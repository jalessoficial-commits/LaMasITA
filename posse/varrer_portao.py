"""Varre a janela do portão (frames antes e depois de cada evento) e compara no GT público.

    python posse/varrer_portao.py                  # bola morta do GT (--live-ha)
    python posse/varrer_portao.py --dead nuno      # bola morta do Nuno (analise/out/paradas_D.csv)
    python posse/varrer_portao.py --janelas 25 30 40 50 75 100  # só janelas simétricas, estas
    python posse/varrer_portao.py --sem-portao     # acrescenta duas linhas sem portão (peso 1,5 e 0,5)
    python posse/varrer_portao.py --extra --touch-weight 0.75   # repassa opções ao preditor

Duas famílias de janela, sempre com a bola campeã (Viterbi do tracking, sem filtro de pulos) e as
demais regras do preditor como estão:

  simétrica    o evento libera a troca de N frames antes a N depois (o padrão é N = 15; era 25)
  só antes     depois fica em 25 e só a parte de ANTES encolhe. As trocas certas saem adiantadas
               (12665 -21, 13785 -19, 17293 -23): com pouca janela antes do toque, a troca não pode
               cair muito antes dele

O ATD é lido uma vez. Cada rodada grava posse/out/portao_<antes>_<depois>.csv ou
posse/out/sem_portao_t<peso>.csv (sobrescritos) e a
camada posse/out/toques.csv fica com a última rodada. Na tabela:

  certas / perdidas     trocas do GT acertadas (até 2 s, mesmo sentido) e perdidas (de 12)
  inventadas            trocas previstas sem par no GT
  bloqueadas            trocas do GT que o portão não liberava
  desvio                mediana do desvio das trocas certas, em frames (negativo = adiantada)
  liberada H / A        fração dos frames vivos em que a troca para cada time está liberada

Com 12 trocas, diferenças de poucos pontos no spellF1 são ruído (uma posse move ~6 pontos com a
bola morta do GT): olhe também certas, inventadas e desvio.
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comparar_bolas as cb  # noqa: E402

GT = os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
NUNO = os.path.join("analise", "out", "paradas_D.csv")
SIMETRICA = (25, 20, 15, 10, 5)
SO_ANTES = (15, 10, 5, 0)          # com 25 depois
LIBERADA = re.compile(r"troca liberada em (\d+)% dos frames vivos para H e (\d+)% para A")


def desvio_mediano(text):
    rows = [cb.ROW.match(line) for line in text.splitlines()]
    offsets = [int(r.group(3).split()[0]) for r in rows if r and "frames" in r.group(3)]
    return float(np.median(offsets)) if offsets else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", choices=("gt", "nuno"), default="gt",
                        help="gt = fronteiras do GT público (--live-ha); nuno = analise/out/paradas_D.csv")
    parser.add_argument("--ball", default=os.path.join("tracking_bola", "out", "viterbi_bola.csv"))
    parser.add_argument("--ball-units", default="px")
    parser.add_argument("--janelas", type=int, nargs="+", default=None,
                        help="janelas simétricas a testar (frames de cada lado); com esta opção, a "
                             "família 'só antes' não roda")
    parser.add_argument("--sem-portao", action="store_true",
                        help="acrescenta duas rodadas sem portão: peso do toque 1,5 (a política "
                             "antiga) e 0,5 (só o portão muda)")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="opções repassadas ao preditor em todas as rodadas (vai por último)")
    args = parser.parse_args()
    os.chdir(cb.ROOT)

    dead = ["--dead", GT, "--live-ha"] if args.dead == "gt" else ["--dead", NUNO]
    if not os.path.isfile(dead[1]):
        raise SystemExit("não achei {}".format(dead[1]))
    if args.janelas:
        windows = [(n, n) for n in args.janelas]
    else:
        windows = [(n, n) for n in SIMETRICA] + [(b, 25) for b in SO_ANTES]
    configs = [("{} / {}".format(b, a), "portao_{}_{}".format(b, a),
                ["--gate-window", str(a), "--gate-before", str(b)]) for b, a in windows]
    if args.sem_portao:
        configs += [("sem portão 1,5", "sem_portao_t15", ["--no-gate"]),
                    ("sem portão 0,5", "sem_portao_t05", ["--no-gate", "--touch-weight", "0.5"])]

    results = []
    for label, name, options in configs:
        print("rodando: {} ...".format(label), flush=True)
        argv = dead + ["--ball", args.ball, "--ball-units", args.ball_units, "--no-despike"] + \
            options + ["--out", os.path.join(HERE, "out", name + ".csv")]
        text = cb.run(argv + args.extra)
        info = cb.parse(text)
        info["janela"] = label
        info["desvio"] = desvio_mediano(text)
        m = LIBERADA.search(text)
        info["lib"] = "{}% / {}%".format(m.group(1), m.group(2)) if m else "-"
        results.append(info)

    print("\nbola morta: {}".format("GT público (--live-ha)" if args.dead == "gt" else NUNO))
    print("{:<16} {:>8} {:>8} {:>7} {:>7} {:>9} {:>11} {:>11} {:>7} {:>15}".format(
        "antes / depois", "spellF1", "macroF1", "posses", "certas", "perdidas", "inventadas",
        "bloqueadas", "desvio", "liberada H / A"))
    for r in results:
        print("{:<16} {:>8.4f} {:>8.4f} {:>7} {:>7} {:>9} {:>11} {:>11} {:>7.0f} {:>15}".format(
            r["janela"], r.get("spell", float("nan")), r.get("macro", float("nan")),
            r.get("posses", "-"), r["certas"], r["perdidas"], r["inventadas"], r["bloqueadas"],
            r["desvio"], r["lib"]))
    print("\nGT público: 17 posses, 12 trocas com a bola viva. A linha 15 / 15 é o padrão do preditor.")


if __name__ == "__main__":
    main()