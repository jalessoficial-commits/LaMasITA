"""Testa árvores com menos árvores e/ou mais rasas: o equivalente a "treinar menos" das redes.

    python -m posse_ml.testar_arvores                          # HGB: 50, 150 e 400 árvores x 15 e 63 folhas
    python -m posse_ml.testar_arvores --arvores 25 50 --folhas 7 15
    python -m posse_ml.testar_arvores --classificador xgboost  # também xgboost e lightgbm
    python -m posse_ml.testar_arvores --so-comparar            # não treina; só refaz a tabela

Cada configuração fica em modelos/<instância>/<classificador>_a<árvores>_f<folhas>/. A validação usa o
mesmo número de árvores do modelo final (sem isso, a validação rápida usa 150 e o modelo final 400,
o mesmo descompasso que atrapalhou as redes); continua rápida só em usar um frame a cada 4. O resto
da configuração é o de sempre (taxa 0,05; no mínimo 200 amostras por folha no HGB). Árvore não
depende de semente: uma rodada por configuração.

A configuração padrão do classificador (400 árvores e 63 folhas, validada com 150) entra como
referência, se existir. Para cada uma: validação na DFL, as três versões na janela, o teto e as
medidas por frame do modelo. A tabela sai na tela e em modelos/<instância>/arvores.md.
"""

import argparse
import importlib.util
import os
import sys

from . import comparar_classificadores as C, config

# nome do número de árvores e do número de folhas em cada classificador
PARAMETROS = {"hgb": ("max_iter", "max_leaf_nodes"), "xgboost": ("n_estimators", "max_leaves"),
              "lightgbm": ("n_estimators", "num_leaves")}


def nome(clf, a, f):
    return "{}_a{}_f{}".format(clf, a, f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificador", choices=list(PARAMETROS), default="hgb")
    ap.add_argument("--arvores", nargs="+", type=int, default=[50, 150, 400])
    ap.add_argument("--folhas", nargs="+", type=int, default=[15, 63])
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst, clf = args.instancia, args.classificador
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para comparar".format(config.GT_PUBLICO))
    if importlib.util.find_spec(C.PACOTE[clf]) is None:
        sys.exit("{}: pacote {} não instalado".format(clf, C.PACOTE[clf]))
    p_arv, p_fol = PARAMETROS[clf]
    combos = [(a, f) for f in args.folhas for a in args.arvores]

    # 1. treino do que falta
    falhas, tempos = [], {}
    for i, (a, f) in enumerate(combos, 1):
        nm = nome(clf, a, f)
        pasta = config.pasta_modelos(inst, nm)
        if C.treinado(pasta) or args.so_comparar:
            print("\n=== [{}/{}] {}: {}".format(i, len(combos), nm,
                                              "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
            continue
        hiper = "{}={},{}={}".format(p_arv, max(2, a // 10) if args.teste else a, p_fol, f)
        if args.teste and clf == "hgb":
            hiper += ",min_samples_leaf=50"
        print("\n=== [{}/{}] {}".format(i, len(combos), nm))
        ok, tempos[nm] = C.treinar_pasta(inst, clf, pasta, hiper, sem_gpu=args.sem_gpu,
                                         passo=3 if args.teste else None, nome=nm)
        if not ok:
            falhas.append(nm)

    # 2. janela pública: a configuração padrão (referência) e as testadas
    versoes = [(clf, None, None)] if C.treinado(config.pasta_modelos(inst, clf)) else []
    versoes += [(nome(clf, a, f), a, f) for a, f in combos if C.treinado(config.pasta_modelos(inst, nome(clf, a, f)))]
    if not versoes:
        sys.exit("nada treinado para comparar")
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    linhas = []
    for nm, a, f in versoes:
        try:
            linhas.append((nm, a, f, C.avaliar(p, x, nomes, plano, gt, inst, nm, paradas, teto=True)))
        except Exception as erro:            # uma configuração com problema não derruba a comparação
            print("=== {}: falhou na comparação: {}".format(nm, erro))
            falhas.append(nm)

    cab = ["modelo", "árvores (validação / final)", "folhas", "treino", "validação DFL (spellF1 / macroF1)",
           "janela, bola morta própria", "janela, detector rígido", "janela, detector suave",
           "teto (bola morta do GT, não vale)", "por frame: morta/viva", "por frame: H/A (bola viva)"]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for nm, a, f, ln in linhas:
        v = ln["validacao"]
        arv = "150 / 400 (padrão)" if a is None else "{0} / {0}".format(a)
        tab.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            nm + (" (referência)" if a is None else ""), arv, 63 if f is None else f, tempos.get(nm, "já treinado"),
            "{} / {}".format(C._num(v[0]), C._num(v[1])) if v[0] else "—",
            C._placar(ln.get("propria")), C._placar(ln.get("rigido")), C._placar(ln.get("suave")),
            C._placar(ln.get("teto")), C._num(ln["frames_morta"]), C._num(ln["frames_ha"])))
    texto = "\n".join(tab)
    print("\n" + texto + "\n\n" + C.NOTA)
    destino = os.path.join(config.pasta_modelos(inst), "arvores.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Árvores ({}), instância {}\n\n{}\n\n{}\n".format(clf, inst, texto, C.NOTA))
    print("tabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
