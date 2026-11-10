"""Treina os mesmos classificadores com outras sementes e mostra se o resultado se repete.

    python -m posse_ml.testar_sementes                              # tcn e gru, sementes 1 e 2
    python -m posse_ml.testar_sementes --classificadores tcn --sementes 1 2 3 4
    python -m posse_ml.testar_sementes --so-comparar                # não treina; só refaz a tabela

A semente 0 é o modelo que já existe (modelos/<instância>/tcn). Cada semente nova muda só o sorteio
do modelo (pesos iniciais da rede, ordem das janelas, dropout), com --hiper semente=N; as cópias
degradadas da DFL são as mesmas, já no cache. Os modelos ficam em modelos/<instância>/<clf>_s<N>/.

Para cada classificador, depois de treinar:
  1. a média das sementes (<clf>_sementes), validada na DFL com as probabilidades do cache, como no
     combinar.py, e gravada como uma combinação;
  2. cada semente e a média calibradas, com as submissões gravadas (posse_ml_<instância>_<clf>_s<N>_...)
     e medidas na janela pública, mais o teto (bola morta do GT, sem gravar arquivo).
A tabela sai na tela e em modelos/<instância>/sementes.md.

Como ler: se todas as sementes de um classificador vão parecido na janela, o resultado é do método;
se se espalham, o melhor número de uma semente foi sorte, e o que se espera é a média delas.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

from . import combinar, comparar_classificadores as C, config


def nome_semente(clf, s):
    return clf if s == 0 else "{}_s{}".format(clf, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificadores", nargs="+", choices=C.ORDEM, default=["tcn", "gru"])
    ap.add_argument("--sementes", nargs="+", type=int, default=[1, 2],
                    help="sementes novas (a 0 é o modelo que já existe e entra sempre)")
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--completo", action="store_true", help="validação inteira, sem --rapido")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst = args.instancia
    sementes = [0] + sorted(set(s for s in args.sementes if s != 0))
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para calibrar e comparar".format(config.GT_PUBLICO))

    # 1. treino das sementes que faltam
    falhas = []
    for clf in args.classificadores:
        if importlib.util.find_spec(C.PACOTE[clf]) is None:
            sys.exit("{}: pacote {} não instalado".format(clf, C.PACOTE[clf]))
        if not C.treinado(config.pasta_modelos(inst, clf)):
            sys.exit("{} (semente 0) não está treinado: rode antes python -m posse_ml.comparar_classificadores "
                     "--classificadores {}".format(clf, clf))
        for s in sementes[1:]:
            nome = nome_semente(clf, s)
            pasta = config.pasta_modelos(inst, nome)
            if C.treinado(pasta) or args.so_comparar:
                print("\n=== {}: {}".format(nome, "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
                continue
            hiper = "semente={}".format(s) + ("," + C.PEQUENOS[clf] if args.teste else "")
            ok, _ = C.treinar_pasta(inst, clf, pasta, hiper, args.completo, args.sem_gpu, 3 if args.teste else None)
            if not ok:
                falhas.append(nome)

    # 2. a média das sementes de cada classificador, validada na DFL
    grupos = {}
    for clf in args.classificadores:
        prontas = [nome_semente(clf, s) for s in sementes if C.treinado(config.pasta_modelos(inst, nome_semente(clf, s)))]
        grupos[clf] = prontas
        if len(prontas) >= 2:
            print("\n=== {}: média das sementes na DFL".format(clf))
            combinar.combinar(inst, prontas, clf + "_sementes")

    # 3. janela pública: cada semente, a média e o teto
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    linhas = []
    for clf in args.classificadores:
        versoes = grupos[clf] + ([clf + "_sementes"] if len(grupos[clf]) >= 2 else [])
        for v in versoes:
            linhas.append(C.avaliar(p, x, nomes, plano, gt, inst, v, paradas, teto=True))

    cab = ["modelo", "validação DFL (spellF1 / macroF1)", "janela, bola morta própria", "janela, detector rígido",
           "janela, detector suave", "teto (bola morta do GT, não vale)"]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for ln in linhas:
        v = ln["validacao"]
        tab.append("| {} | {} | {} | {} | {} | {} |".format(
            ln["clf"], "{} / {}".format(C._num(v[0]), C._num(v[1])) if v[0] else "—",
            C._placar(ln.get("propria")), C._placar(ln.get("rigido")), C._placar(ln.get("suave")),
            C._placar(ln.get("teto"))))
    resumo = []
    for clf in args.classificadores:
        sozinhas = [ln for ln in linhas if ln["clf"] in grupos[clf]]
        for col in ("rigido", "suave", "teto"):
            vals = [ln[col][0] for ln in sozinhas if ln.get(col)]
            if len(vals) >= 2:
                resumo.append("{} {}: spellF1 na janela de {} a {} (média {}, {} sementes)".format(
                    clf, col, C._num(min(vals)), C._num(max(vals)), C._num(float(np.mean(vals))), len(vals)))
        dfl = [ln["validacao"][0] for ln in sozinhas if ln["validacao"][0]]
        if len(dfl) >= 2:
            resumo.append("{} DFL: spellF1 de {} a {} (média {})".format(
                clf, C._num(min(dfl)), C._num(max(dfl)), C._num(float(np.mean(dfl)))))
    texto = "\n".join(tab) + "\n\n" + "\n".join("- " + r for r in resumo)
    print("\n" + texto + "\n\n" + C.NOTA)
    destino = os.path.join(config.pasta_modelos(inst), "sementes.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Sementes, instância {}\n\n{}\n\n{}\n".format(inst, texto, C.NOTA))
    print("tabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
