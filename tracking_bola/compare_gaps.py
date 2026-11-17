"""Compara, frame a frame, dois trackers onde o ATD NÃO tem bola (seção B do compare_trackers), com recortes.

Colocar em tracking_bola/, junto de compare_trackers.py, eval_atd.py e check_projection.py. Precisa de
opencv-python. Padrão: campeão (out/full_bola.csv) x Viterbi (out/viterbi_bola.csv) no conjunto de TESTE:

    python tracking_bola/compare_gaps.py
    python tracking_bola/compare_gaps.py --annotations tracking_bola/out/annotations/ball_annotations.csv   # os 292 frames

Parte 1 - frames em que você VÊ a bola e o ATD não tem bola (o "n = 29" da seção B no teste). Para cada um:
posição e erro de cada tracker, o resultado (acerto / longe / sem bola) e a regra que o Viterbi usou
(detector, interpolado...). No fim, a tabela cruzada: os dois acertam, só o Viterbi, só o campeão, nenhum.

Parte 2 - frames em que você NÃO vê a bola e o ATD também não: o que cada tracker diz (é onde nasce o falso
alarme a mais do Viterbi), com a regra do Viterbi.

Recortes em out/gap_analysis/ (um por frame e uma folha com todos): verde = seu clique, vermelho = campeão,
amarelo = Viterbi. São recortes do vídeo: não publique nem suba para o Git.
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

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (40, 40, 255), (0, 230, 255), (255, 255, 255)


def read_rules(path):
    """{frame: regra} se o CSV tiver a coluna 'regra'; senão {}."""
    rules = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "regra" not in (reader.fieldnames or []):
            return rules
        for row in reader:
            rules[int(float(row["frame"]))] = row["regra"]
    return rules


def outcome(point, truth, tol):
    if point is None:
        return "sem bola", None
    error = float(np.hypot(point[0] - truth[0], point[1] - truth[1]))
    return ("acerto" if error <= tol else "longe"), error


def crop(cv2, image, center, half, zoom):
    height, width = image.shape[:2]
    cx = int(min(max(center[0], half), width - half))
    cy = int(min(max(center[1], half), height - half))
    piece = image[cy - half:cy + half, cx - half:cx + half]
    return cv2.resize(piece, (2 * half * zoom, 2 * half * zoom), interpolation=cv2.INTER_NEAREST)


def sheet(tiles, columns):
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    out = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", default=os.path.join(HERE, "out", "annotations", "ball_annotations_teste.csv"))
    parser.add_argument("--a", default=os.path.join(HERE, "out", "full_bola.csv"), help="tracker A (padrão: campeão)")
    parser.add_argument("--b", default=os.path.join(HERE, "out", "viterbi_bola.csv"), help="tracker B (padrão: Viterbi)")
    parser.add_argument("--name-a", default="campeao")
    parser.add_argument("--name-b", default="viterbi")
    parser.add_argument("--tol", type=float, default=40.0)
    parser.add_argument("--half", type=int, default=100, help="meio lado do recorte, px (padrão 100)")
    parser.add_argument("--zoom", type=int, default=3)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "out", "gap_analysis"))
    args = parser.parse_args(argv)
    import cv2

    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, _, valid = load_homography()
    best_a, _, _ = load_tracker(args.a)
    best_b, _, _ = load_tracker(args.b)
    rules_b = read_rules(args.b)
    no_atd = {f: v for f, v in annotations.items()
              if 0 <= f < n_frames and atd_pixel(ball, matrices, valid, f) is None}
    seen = sorted(f for f, (s, _, _) in no_atd.items() if s == "ball")
    hidden = sorted(f for f, (s, _, _) in no_atd.items() if s == "not_visible")

    os.makedirs(args.out_dir, exist_ok=True)
    for name in os.listdir(args.out_dir):
        if name.endswith(".jpg"):
            os.remove(os.path.join(args.out_dir, name))
    capture = cv2.VideoCapture(require(["video"])[0])
    tiles = []

    def picture(frame, truth, pa, pb, label):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            return
        marked = image.copy()
        for point, color, radius in ((truth, GREEN, 18), (pa, RED, 13), (pb, YELLOW, 8)):
            if point is not None:
                cv2.circle(marked, (int(round(point[0])), int(round(point[1]))), radius, color, 2)
        center = truth or pb or pa or (image.shape[1] / 2, image.shape[0] / 2)
        tile = crop(cv2, marked, center, args.half, args.zoom)
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.imwrite(os.path.join(args.out_dir, "frame_{:06d}.jpg".format(frame)), marked)
        tiles.append(tile)

    print("PARTE 1: você vê a bola e o ATD não tem bola (n = {})".format(len(seen)))
    print("{:>7}  {:<22} {:<22} {:<12}".format("frame", args.name_a, args.name_b, "regra " + args.name_b))
    cross = {}
    for frame in seen:
        truth = annotations[frame][1:]
        pa, pb = best_a.get(frame), best_b.get(frame)
        pa = pa[:2] if pa else None
        pb = pb[:2] if pb else None
        ra, ea = outcome(pa, truth, args.tol)
        rb, eb = outcome(pb, truth, args.tol)
        key = (ra == "acerto", rb == "acerto")
        cross[key] = cross.get(key, 0) + 1
        text_a = ra if ea is None else "{} ({:.0f} px)".format(ra, ea)
        text_b = rb if eb is None else "{} ({:.0f} px)".format(rb, eb)
        print("{:>7}  {:<22} {:<22} {:<12}".format(frame, text_a, text_b, rules_b.get(frame, "-")))
        picture(frame, truth, pa, pb, "{} | {}: {} | {}: {}".format(frame, args.name_a, ra, args.name_b, rb))
    total = max(1, len(seen))
    print("\n  os dois acertam      {:3d} ({:5.1f}%)".format(cross.get((True, True), 0), 100.0 * cross.get((True, True), 0) / total))
    print("  só o {:<14} {:3d} ({:5.1f}%)".format(args.name_b, cross.get((False, True), 0), 100.0 * cross.get((False, True), 0) / total))
    print("  só o {:<14} {:3d} ({:5.1f}%)".format(args.name_a, cross.get((True, False), 0), 100.0 * cross.get((True, False), 0) / total))
    print("  nenhum acerta        {:3d} ({:5.1f}%)".format(cross.get((False, False), 0), 100.0 * cross.get((False, False), 0) / total))
    rules_hit = {}
    for frame in seen:
        pb = best_b.get(frame)
        if pb is not None and outcome(pb[:2], annotations[frame][1:], args.tol)[0] == "acerto":
            rule = rules_b.get(frame, "-")
            rules_hit[rule] = rules_hit.get(rule, 0) + 1
    print("  acertos do {} por regra: {}".format(args.name_b, rules_hit or "-"))

    print("\nPARTE 2: você NÃO vê a bola e o ATD também não tem (n = {})".format(len(hidden)))
    print("{:>7}  {:<12} {:<12} {:<12}".format("frame", args.name_a, args.name_b, "regra " + args.name_b))
    for frame in hidden:
        pa, pb = best_a.get(frame), best_b.get(frame)
        say_a = "dá bola" if pa else "sem bola"
        say_b = "dá bola" if pb else "sem bola"
        print("{:>7}  {:<12} {:<12} {:<12}".format(frame, say_a, say_b, rules_b.get(frame, "-")))
        picture(frame, None, pa[:2] if pa else None, pb[:2] if pb else None,
                "{} SEM BOLA | {}: {} | {}: {}".format(frame, args.name_a, say_a, args.name_b, say_b))

    out = sheet(tiles, 4)
    if out is not None:
        cv2.imwrite(os.path.join(args.out_dir, "contact_sheet.jpg"), out)
    print("\nRecortes em {} (verde = seu clique, vermelho = {}, amarelo = {}).".format(args.out_dir, args.name_a, args.name_b))


if __name__ == "__main__":
    main()