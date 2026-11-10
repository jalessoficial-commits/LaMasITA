"""Compara o número de épocas do modelo final, com várias sementes, sempre validando com 5 épocas.

    python -m posse_ml.testar_epocas                      # lstm, transformer e tcn; 5, 10 e 20 épocas; sementes 0, 1, 2
    python -m posse_ml.testar_epocas --classificadores tcn --epocas 5 10 --sementes 0 1
    python -m posse_ml.testar_epocas --so-comparar        # não treina; só refaz a tabela

Cada combinação (rede, épocas, semente) é um modelo, em modelos/<instância>/:

  <rede>_f<N>[_s<semente>]   N épocas no modelo final (f5, f20)
  <rede>[_s<semente>]        10 épocas: os mesmos nomes do testar_sementes e do testar_variantes, que
                             são reaproveitados quando já existem

A validação (que escolhe os custos do Viterbi) é sempre a rápida, com 5 épocas; muda só o modelo
final. Por isso a validação de uma semente é a mesma para 5, 10 e 20 épocas e sai uma vez só, do
cache, e a coluna da DFL não separa as épocas: quem compara as épocas é a janela pública, aqui com a
média de várias sementes, e as medidas por frame do modelo (bola morta/viva e H/A), que oscilam bem
menos que o spellF1 de 17 posses.

Saem, para cada rede e número de épocas, os valores de cada semente e a média deles (não a média
das redes). A tabela fica na tela e em modelos/<instância>/epocas.md; cada modelo também grava as
suas três submissões (posse_ml_<instância>_<nome>_...).
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

from . import comparar_classificadores as C, config

REDES = ("lstm", "transformer", "tcn", "gru")
COLUNAS = (("rigido", "janela, detector rígido"), ("suave", "janela, detector suave"),
           ("teto", "teto (GT, não vale)"), ("propria", "janela, bola morta própria"))


def nome(clf, n, s):
    base = clf if n == 10 else "{}_f{}".format(clf, n)
    return base if s == 0 else "{}_s{}".format(base, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificadores", nargs="+", choices=REDES, default=["lstm", "transformer", "tcn"])
    ap.add_argument("--epocas", nargs="+", type=int, default=[5, 10, 20], help="épocas do modelo final")
    ap.add_argument("--sementes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst = args.instancia
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para comparar".format(config.GT_PUBLICO))
    if importlib.util.find_spec("torch") is None:
        sys.exit("as redes precisam do PyTorch (requirements-classificadores.txt)")
    combos = [(c, n, s) for c in args.classificadores for n in args.epocas for s in args.sementes]

    # 1. treino do que falta; a validação de cada (rede, semente) é a mesma para todas as épocas
    falhas, tempos = [], {}
    for i, (clf, n, s) in enumerate(combos, 1):
        nm = nome(clf, n, s)
        pasta = config.pasta_modelos(inst, nm)
        if C.treinado(pasta) or args.so_comparar:
            print("\n=== [{}/{}] {}: {}".format(i, len(combos), nm,
                                              "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
            continue
        if args.teste:
            hiper = "semente={},epocas={},epocas_rapido=1".format(s, max(1, n // 5))
        else:
            hiper = "semente={},epocas={},epocas_rapido=5".format(s, n)
        print("\n=== [{}/{}] {}".format(i, len(combos), nm))
        ok, tempos[nm] = C.treinar_pasta(inst, clf, pasta, hiper, sem_gpu=args.sem_gpu,
                                         passo=3 if args.teste else None, nome=nm)
        if not ok:
            falhas.append(nm)

    # 2. janela pública, com o jogo lido uma vez
    prontos = [(c, n, s) for c, n, s in combos if C.treinado(config.pasta_modelos(inst, nome(c, n, s)))]
    if not prontos:
        sys.exit("nada treinado para comparar")
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    res = {}
    for c, n, s in prontos:
        try:
            res[(c, n, s)] = C.avaliar(p, x, nomes, plano, gt, inst, nome(c, n, s), paradas, teto=True)
        except Exception as erro:            # um modelo com problema não derruba a comparação
            print("=== {}: falhou na comparação: {}".format(nome(c, n, s), erro))
            falhas.append(nome(c, n, s))

    # 3. tabelas: resumo por (rede, épocas) com a média das sementes, e o detalhe de cada modelo
    def valor(ln, col):
        if col in ("frames_morta", "frames_ha"):
            return ln.get(col)
        if col == "dfl":
            return ln["validacao"][0]
        return ln[col][0] if ln.get(col) else None

    def celula(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        return "**{}** ({})".format(C._num(float(np.mean(vals))), " / ".join(C._num(v) for v in vals))

    cols = [("dfl", "DFL (validação, 5 épocas)")] + list(COLUNAS) + [
        ("frames_morta", "por frame: morta/viva"), ("frames_ha", "por frame: H/A (bola viva)")]
    cab = ["rede", "épocas", "sementes"] + [t for _, t in cols]
    resumo = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for c in args.classificadores:
        for n in args.epocas:
            lns = [res[(c, n, s)] for s in args.sementes if (c, n, s) in res]
            if not lns:
                continue
            resumo.append("| {} | {} | {} | {} |".format(c, n, len(lns), " | ".join(
                celula([valor(ln, k) for ln in lns]) for k, _ in cols)))

    cab2 = ["modelo", "rede", "épocas", "semente", "treino", "DFL", "rígido", "suave", "teto", "própria",
            "morta/viva por frame", "H/A por frame"]
    detalhe = ["| " + " | ".join(cab2) + " |", "|" + "---|" * len(cab2)]
    for (c, n, s), ln in res.items():
        nm = nome(c, n, s)
        detalhe.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            nm, c, n, s, tempos.get(nm, "já treinado"), C._num(ln["validacao"][0]) if ln["validacao"][0] else "—",
            C._placar(ln.get("rigido")), C._placar(ln.get("suave")), C._placar(ln.get("teto")),
            C._placar(ln.get("propria")), C._num(ln["frames_morta"]), C._num(ln["frames_ha"])))

    nota = ("Cada célula: **média das sementes** (valor de cada semente, na ordem). Janela: spellF1. A DFL é a "
            "mesma para todas as épocas da mesma semente (a validação usa sempre 5 épocas). Na tabela de "
            "detalhe, janela: spellF1 / macroF1 (posses previstas; GT 17).")
    texto = "## Resumo\n\n" + "\n".join(resumo) + "\n\n" + nota + "\n\n## Cada modelo\n\n" + "\n".join(detalhe)
    print("\n" + texto)
    destino = os.path.join(config.pasta_modelos(inst), "epocas.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Épocas, instância {}\n\n{}\n".format(inst, texto))
    print("\ntabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
