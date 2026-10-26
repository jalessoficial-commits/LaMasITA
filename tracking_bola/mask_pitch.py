"""Remove candidatos de bola que caem fora do campo (além de uma margem), usando a homografia de cada frame.

Colocar em tracking_bola/. Só numpy e pandas (os mesmos do load_homography):

    python tracking_bola/mask_pitch.py --detector tracking_bola/out/yolo_roboflow.csv --out tracking_bola/out/yolo_roboflow_campo.csv
    python tracking_bola/mask_pitch.py --detector tracking_bola/out/wasb_060.csv --margin 2 --out tracking_bola/out/wasb_060_campo.csv

Depois, o CSV filtrado entra no lugar do original na fusão e na comparação:

    python tracking_bola/fuse_atd_wasb.py --wasb tracking_bola/out/yolo_roboflow_campo.csv --solo-conf 0.8 --override-conf 0.8 --out tracking_bola/out/fusao_rf_campo.csv

Como funciona: a homografia do frame leva coordenadas normalizadas do campo (0-1, as mesmas do ATD) para
pixels; a inversa leva cada candidato (pixel) de volta ao campo. O candidato fica se cair dentro do
retângulo do campo ampliado por --margin metros de cada lado (padrão 2 m). As dimensões do campo vêm
de games.py (medidas pela organização); se não estiverem lá, usa 105 x 68 m.

Por que a margem: a bola de verdade sai do campo (lateral, tiro de meta, escanteio) e em geral fica a
poucos metros da linha. Cuidado com bola alta: a homografia vale para o plano do chão, então uma bola no
ar é projetada mais longe do que está e pode cair "fora". Frames sem homografia válida não são filtrados.

Saída: mesmo formato da entrada (frame,x,y[,conf]), só com os candidatos dentro do campo. Como cada
frame pode ter vários candidatos, tirar a bola reserva deixa o próximo candidato (muitas vezes a bola
de verdade) assumir como o de maior confiança.
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
import games  # noqa: E402
from games import GAME, source_frames  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402

DEFAULT_PITCH = (105.0, 68.0)


def pitch_size():
    """(comprimento, largura) em metros: games.GAMES[GAME]['pitch'] se existir, senão 105 x 68."""
    try:
        length, width = games.GAMES[GAME]["pitch"]
        return float(length), float(width)
    except (AttributeError, KeyError, TypeError, ValueError):
        return DEFAULT_PITCH


def inside(u, v, margin_u, margin_v):
    return (-margin_u <= u <= 1.0 + margin_u) and (-margin_v <= v <= 1.0 + margin_v)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detector", required=True, metavar="CSV", help="CSV de candidatos (frame,x,y[,conf])")
    parser.add_argument("--margin", type=float, default=2.0, help="margem fora das linhas, em metros (padrão 2)")
    parser.add_argument("--out", required=True, metavar="CSV", help="CSV filtrado")
    args = parser.parse_args(argv)
    if os.path.abspath(args.out) == os.path.abspath(args.detector):
        parser.error("--out não pode ser o mesmo arquivo de --detector")
    if not os.path.isfile(args.detector):
        sys.exit("não achei {}".format(args.detector))

    length, width = pitch_size()
    margin_u, margin_v = args.margin / length, args.margin / width
    matrices, _, valid = load_homography()
    n_frames = min(len(matrices), source_frames("tactical", GAME))
    inverses = {}

    with open(args.detector, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        try:
            col_frame, col_x, col_y = header.index("frame"), header.index("x"), header.index("y")
        except ValueError:
            sys.exit("{}: o cabeçalho precisa ter frame, x e y".format(args.detector))
        rows = [row for row in reader if row]

    kept, removed, unchecked = [], 0, 0
    frames_before, frames_after, removed_points = set(), set(), []
    for row in rows:
        frame = int(float(row[col_frame]))
        x, y = float(row[col_x]), float(row[col_y])
        frames_before.add(frame)
        if not (0 <= frame < n_frames) or not valid[frame]:
            kept.append(row)
            frames_after.add(frame)
            unchecked += 1
            continue
        if frame not in inverses:
            inverses[frame] = np.linalg.inv(matrices[frame])
        u, v = project(inverses[frame], x, y)
        if inside(float(u), float(v), margin_u, margin_v):
            kept.append(row)
            frames_after.add(frame)
        else:
            removed += 1
            removed_points.append((float(u) * length, float(v) * width))

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(kept)

    print("campo {:.1f} x {:.1f} m, margem {:g} m".format(length, width, args.margin))
    print("candidatos: {} -> {} ({} removidos fora do campo)".format(len(rows), len(kept), removed))
    print("frames com algum candidato: {} -> {}".format(len(frames_before), len(frames_after)))
    if unchecked:
        print("{} candidatos em frames sem homografia válida ficaram sem filtro".format(unchecked))
    if removed_points:
        points = np.array(removed_points)
        outside_x = np.maximum(-points[:, 0], points[:, 0] - length)
        outside_y = np.maximum(-points[:, 1], points[:, 1] - width)
        beyond = np.maximum(outside_x, outside_y)
        print("distância dos removidos até a linha: mediana {:.1f} m, mínimo {:.1f} m".format(
            float(np.median(beyond)), float(beyond.min())))
    print("saída: {}".format(args.out))


if __name__ == "__main__":
    main()