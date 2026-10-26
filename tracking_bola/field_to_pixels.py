"""Converte um tracker de bola em coordenadas do CAMPO (0-1, como o ATD) para pixels, para avaliar no compare_trackers.

Colocar em tracking_bola/, junto de compare_trackers.py. Só numpy e pandas (os do load_homography):

    python tracking_bola/field_to_pixels.py --tracker data/tactical/tracker_kalman_pred.csv
    python tracking_bola/compare_trackers.py --tracker campeao=tracking_bola/out/full_bola.csv --tracker kalman_colega=tracking_bola/out/tracker_kalman_pred_px.csv --tol 40

Entrada: CSV com colunas frame, x, y (x, y normalizados no campo; outras colunas são ignoradas; linha sem
x/y = sem bola). Saída: tracking_bola/out/<nome>_px.csv com frame,x,y em pixels do vídeo, pela homografia
de cada frame.

Índice do frame: o ATD conta a partir de 1 e o vídeo a partir de 0. Para não errar por um frame, o script
compara o arquivo com a bola do nosso tracker (data/tactical/BAR-ATH_tactical_ball_tracker.csv) deslocando
o índice em -1, 0 e +1 e mostra a distância mediana em cada caso; o deslocamento certo é o de menor
distância (os dois trackers vêm do mesmo detector, então devem coincidir em muitos frames). Use
--frame-offset para aplicar outro deslocamento (padrão: o escolhido automaticamente).

Também mostra quanto do arquivo cai muito fora do campo e quantos saltos fisicamente impossíveis
(> --vmax m/s entre frames seguidos) ele tem. DADO DERIVADO DO VÍDEO: a saída fica em out/, fora do Git.
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
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from mask_pitch import pitch_size  # noqa: E402


def read_field(path):
    """{frame: (x, y)} das linhas com x e y."""
    result = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for column in ("frame", "x", "y"):
            if column not in (reader.fieldnames or []):
                sys.exit("{}: falta a coluna '{}'".format(path, column))
        for row in reader:
            if row["x"] in ("", "nan", "NaN") or row["y"] in ("", "nan", "NaN"):
                continue
            result[int(float(row["frame"]))] = (float(row["x"]), float(row["y"]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", required=True, help="CSV em coordenadas do campo (frame,x,y)")
    parser.add_argument("--reference", default=None, help="padrão: data/tactical/<jogo>_tactical_ball_tracker.csv")
    parser.add_argument("--frame-offset", type=int, default=None, help="soma isto ao frame do arquivo (padrão: automático)")
    parser.add_argument("--vmax", type=float, default=45.0, help="velocidade máxima plausível, m/s (padrão 45)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    data = read_field(args.tracker)
    n_frames = source_frames("tactical", GAME)
    length, width = pitch_size()
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    reference_path = args.reference or os.path.join(os.path.dirname(require(["atd_raw"])[0]),
                                                    "{}_tactical_ball_tracker.csv".format(GAME))
    offset = args.frame_offset
    if os.path.isfile(reference_path):
        reference = read_field(reference_path)
        print("distância (m) até a bola do nosso tracker, por deslocamento do índice do frame:")
        best = None
        for shift in (-1, 0, 1):
            d = [np.hypot((x - reference[f + shift][0]) * length, (y - reference[f + shift][1]) * width)
                 for f, (x, y) in data.items() if f + shift in reference]
            if d:
                median, same = float(np.median(d)), 100.0 * np.mean(np.array(d) < 0.05)
                print("  {:+d}: mediana {:.3f} m, idênticos (< 5 cm) {:.1f}% de {} frames".format(shift, median, same, len(d)))
                if best is None or median < best[1]:
                    best = (shift, median)
        if offset is None and best is not None:
            offset = best[0]
            print("  -> usando deslocamento {:+d}".format(offset))
    elif offset is None:
        print("sem {} para conferir o índice; usando deslocamento 0".format(reference_path))
    offset = offset or 0

    shifted = {f + offset: xy for f, xy in data.items() if 0 <= f + offset < n_frames}
    frames = np.array(sorted(shifted))
    xy = np.array([shifted[f] for f in frames])
    in_graded = graded[frames]
    print("\nlinhas com bola: {}; nos frames avaliados: {} ({:.1f}% de {})".format(
        len(frames), int(in_graded.sum()), 100.0 * in_graded.sum() / graded.sum(), int(graded.sum())))
    beyond = np.maximum(np.maximum(-xy[:, 0] * length, (xy[:, 0] - 1) * length),
                        np.maximum(-xy[:, 1] * width, (xy[:, 1] - 1) * width))
    print("fora do campo: > 2 m em {:.1f}% das linhas, > 10 m em {:.1f}%".format(
        100.0 * np.mean(beyond > 2), 100.0 * np.mean(beyond > 10)))
    consecutive = np.diff(frames) == 1
    speed = np.hypot(np.diff(xy[:, 0]) * length, np.diff(xy[:, 1]) * width) * 25.0
    print("saltos acima de {:g} m/s entre frames seguidos: {} de {}".format(
        args.vmax, int((speed[consecutive] > args.vmax).sum()), int(consecutive.sum())))

    matrices, _, valid = load_homography()
    name = os.path.splitext(os.path.basename(args.tracker))[0]
    out = args.out or os.path.join(HERE, "out", "{}_px.csv".format(name))
    folder = os.path.dirname(out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    written = 0
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y"])
        for frame, (x, y) in zip(frames, xy):
            if not valid[frame]:
                continue
            px, py = project(matrices[frame], x, y)
            writer.writerow([int(frame), "{:.2f}".format(float(px)), "{:.2f}".format(float(py))])
            written += 1
    print("\n{} linhas em pixels gravadas em {}".format(written, out))


if __name__ == "__main__":
    main()