"""Combina a bola do ATD com os candidatos do WASB (ou de outro detector) e grava um tracker novo.

Colocar em tracking_bola/, junto de compare_trackers.py (este script importa dele). Só numpy:

    python tracking_bola/fuse_atd_wasb.py --wasb tracking_bola/out/wasb_060.csv
    python tracking_bola/compare_trackers.py --tracker wasb=tracking_bola/out/wasb_060.csv --tracker fusao=tracking_bola/out/fusao.csv

A regra, para cada frame (candidatos do detector = linhas do CSV com conf >= --floor):

  1. O ATD NÃO tem bola      -> usa o melhor candidato do detector, se a conf dele for >= --solo-conf
                                (senão, frame sem bola).
  2. O ATD tem bola (a):
     2a. há candidato do detector a até --radius px de a -> usa o de MAIOR conf entre esses
         ("quando os dois concordam, vale a posição do detector, que é mais precisa").
     2b. não há, e o melhor candidato tem conf >= --override-conf -> usa o melhor candidato
         ("quando divergem, mas o detector tem muita certeza, ele vence").
     2c. caso contrário -> usa a posição do ATD.

Com o padrão --override-conf 2 (impossível, conf vai até 1) a 2b nunca acontece: é exatamente a regra
"concordou, vale o detector; divergiu, vale o ATD". Baixe --override-conf (0.7, 0.5...) para deixar o
detector ganhar nas divergências. Dica: --radius deve refletir o erro TÍPICO DO ATD (dezenas de px),
não o do detector; com um raio muito pequeno quase nada "concorda" e a regra vira "sempre o ATD".

Quais frames: os anotados (ball, not_visible, uncertain), como nos detectores. O resultado é um CSV
frame,x,y (sem conf), em pixels do vídeo original, no formato que o compare_trackers.py lê.
Cada frame que sai com bola traz também, no terminal, de que regra veio (resumo no final).

Cuidado com o ajuste: --radius, --override-conf e --solo-conf são escolhidos olhando os mesmos frames em
que você mede o resultado, então o número sai otimista. Poucos parâmetros ajudam; o ideal é medir o
resultado final em frames que você não usou para escolher os valores.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require, source_frames  # noqa: E402
from load_homography import load as load_homography  # noqa: E402
from check_projection import read_ball  # noqa: E402
from eval_atd import read_annotations, atd_pixel  # noqa: E402
from compare_trackers import load_tracker  # noqa: E402

DEFAULT_ANNOTATIONS = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
DEFAULT_OUT = os.path.join(HERE, "out", "fusao.csv")
STATUSES = ("ball", "not_visible", "uncertain")


def fuse_frame(atd_xy, candidates, radius, override_conf, solo_conf, floor):
    """((x, y) ou None, regra) de um frame. candidates = [(x, y, conf), ...] do detector."""
    usable = sorted((c for c in candidates if c[2] >= floor), key=lambda c: -c[2])
    if atd_xy is None:
        if usable and usable[0][2] >= solo_conf:
            return (usable[0][0], usable[0][1]), "1 só detector"
        return None, "sem bola"
    near = [c for c in usable if np.hypot(c[0] - atd_xy[0], c[1] - atd_xy[1]) <= radius]
    if near:
        return (near[0][0], near[0][1]), "2a concordam: detector"
    if usable and usable[0][2] >= override_conf:
        return (usable[0][0], usable[0][1]), "2b divergem: detector confiante"
    return atd_xy, "2c divergem: ATD"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wasb", required=True, metavar="CSV", help="CSV do detector (frame,x,y,conf)")
    parser.add_argument("--radius", type=float, default=40.0, help="raio de concordância, px (padrão 40)")
    parser.add_argument("--override-conf", type=float, default=2.0,
                        help="conf para o detector vencer quando diverge (padrão 2 = nunca)")
    parser.add_argument("--solo-conf", type=float, default=0.5,
                        help="conf mínima do detector onde o ATD não tem bola (padrão 0.5)")
    parser.add_argument("--floor", type=float, default=0.2, help="descarta candidatos com conf menor (padrão 0.2)")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    frames = sorted(f for f, (s, _, _) in annotations.items() if s in STATUSES and 0 <= f < n_frames)
    _, _, every = load_tracker(args.wasb)
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, _, valid = load_homography()

    rows, counts = [], {}
    for frame in frames:
        point, rule = fuse_frame(atd_pixel(ball, matrices, valid, frame), every.get(frame, []),
                                 args.radius, args.override_conf, args.solo_conf, args.floor)
        counts[rule] = counts.get(rule, 0) + 1
        if point is not None:
            rows.append((frame, point[0], point[1]))

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y"])
        for frame, x, y in rows:
            writer.writerow([frame, "{:.2f}".format(x), "{:.2f}".format(y)])
    print("{} frames; com bola na saída: {}".format(len(frames), len(rows)))
    for rule in sorted(counts):
        print("  {:<34} {:5d}".format(rule, counts[rule]))
    print("saída: {}".format(args.out))


if __name__ == "__main__":
    main()