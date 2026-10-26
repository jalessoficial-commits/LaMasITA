"""Mostra os frames que você marcou como "not_visible" em que um tracker (por padrão, a fusão campeã) deu bola.

Colocar em tracking_bola/, junto de compare_trackers.py, eval_atd.py e check_projection.py (importa deles).
Precisa de opencv-python:

    python tracking_bola/show_false_alarms.py
    python tracking_bola/show_false_alarms.py --tracker tracking_bola/out/fusao_wasb.csv --detector tracking_bola/out/wasb_060.csv

Para cada um desses frames:
  - imprime uma linha: posição que o tracker deu, de onde ela veio (ATD tinha bola ou o detector agiu sozinho),
    a confiança do candidato do detector naquele ponto e a distância até a bola do ATD;
  - salva em out/false_alarms/ o frame inteiro marcado e um recorte ampliado em volta da posição dada;
  - monta out/false_alarms/contact_sheet.jpg com todos os recortes lado a lado.

Marcas no desenho:
  vermelho  posição dada pelo tracker (o falso alarme)
  azul      bola do ATD projetada com a homografia (se o ATD tinha bola)
  amarelo   outros candidatos do detector (os de maior confiança), com a confiança escrita ao lado

No final, agrupa as posições que se repetem (a menos de --cluster px). Se várias caem no mesmo lugar, é
provável um objeto fixo (bola reserva, marca do campo, logo) e uma máscara resolveria. Olhe também os
recortes: pode ser que a bola esteja mesmo lá (escondida em parte, pequena) e a sua anotação é que a perdeu.

ESTAS IMAGENS SÃO RECORTES DO VÍDEO: ficam em out/, que não vai para o Git, e não devem ser publicadas.
"""

import argparse
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
DEFAULT_TRACKER = os.path.join(HERE, "out", "fusao_rf_campo.csv")
DEFAULT_DETECTOR = os.path.join(HERE, "out", "yolo_roboflow.csv")
DEFAULT_OUT_DIR = os.path.join(HERE, "out", "false_alarms")
RED, BLUE, YELLOW, WHITE = (0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 255, 255)


def nearest(candidates, point):
    """(conf, distância) do candidato mais próximo de point; (None, None) se não houver candidatos."""
    if not candidates:
        return None, None
    best = min(candidates, key=lambda c: np.hypot(c[0] - point[0], c[1] - point[1]))
    return best[2], float(np.hypot(best[0] - point[0], best[1] - point[1]))


def clusters(points, radius):
    """Agrupa pontos (frame, x, y) que caem a menos de radius do primeiro ponto do grupo. Lista de grupos."""
    groups = []
    for frame, x, y in points:
        for group in groups:
            if np.hypot(group["x"] - x, group["y"] - y) <= radius:
                group["frames"].append(frame)
                break
        else:
            groups.append({"x": x, "y": y, "frames": [frame]})
    return sorted(groups, key=lambda g: -len(g["frames"]))


def draw(image, fused, atd, candidates, top):
    import cv2
    marked = image.copy()
    for x, y, conf in sorted(candidates, key=lambda c: -c[2])[:top]:
        cv2.circle(marked, (int(round(x)), int(round(y))), 10, YELLOW, 1)
        cv2.putText(marked, "{:.2f}".format(conf), (int(x) + 12, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1)
    if atd is not None:
        cv2.circle(marked, (int(round(atd[0])), int(round(atd[1]))), 16, BLUE, 2)
    cv2.circle(marked, (int(round(fused[0])), int(round(fused[1]))), 14, RED, 2)
    return marked


def crop_zoom(image, center, half, zoom):
    """Recorte de (2*half)x(2*half) em volta de center, ampliado zoom vezes (bordas preenchidas de preto)."""
    import cv2
    height, width = image.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))
    canvas = np.zeros((2 * half, 2 * half, 3), dtype=image.dtype)
    x0, y0 = cx - half, cy - half
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(width, x0 + 2 * half), min(height, y0 + 2 * half)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (2 * half * zoom, 2 * half * zoom), interpolation=cv2.INTER_NEAREST)


def contact_sheet(tiles, columns):
    import cv2
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
        cv2.rectangle(sheet, (c * w, r * h), ((c + 1) * w - 1, (r + 1) * h - 1), WHITE, 1)
    return sheet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="CSV do tracker a inspecionar (padrão: fusao_rf_08_ov.csv)")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR,
                        help="CSV do detector com conf, para mostrar candidatos e confiança (padrão: yolo_roboflow.csv; '' = nenhum)")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--half", type=int, default=60, help="meio lado do recorte, px do vídeo (padrão 60)")
    parser.add_argument("--zoom", type=int, default=3, help="ampliação do recorte (padrão 3)")
    parser.add_argument("--top", type=int, default=5, help="quantos candidatos do detector desenhar (padrão 5)")
    parser.add_argument("--cluster", type=float, default=20.0, help="raio para agrupar posições repetidas, px (padrão 20)")
    parser.add_argument("--columns", type=int, default=6, help="colunas da folha de recortes (padrão 6)")
    args = parser.parse_args(argv)

    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  python -m pip install opencv-python")
    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    hidden = sorted(f for f, (s, _, _) in annotations.items() if s == "not_visible" and 0 <= f < n_frames)
    best, _, _ = load_tracker(args.tracker)
    candidates = load_tracker(args.detector)[2] if args.detector else {}
    alarms = [f for f in hidden if f in best]
    print("{} frames 'not_visible'; o tracker deu bola em {} deles ({:.1f}%)".format(
        len(hidden), len(alarms), 100.0 * len(alarms) / max(1, len(hidden))))
    if not alarms:
        return

    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, _, valid = load_homography()
    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    os.makedirs(args.out_dir, exist_ok=True)
    # apaga as imagens de execuções anteriores (só os arquivos que este script cria), para não misturar
    for name in os.listdir(args.out_dir):
        if (name.startswith(("frame_", "crop_")) and name.endswith(".jpg")) or name == "contact_sheet.jpg":
            os.remove(os.path.join(args.out_dir, name))

    print("\n{:>7} {:>8} {:>8}  {:<20} {:>9} {:>10}".format("frame", "x", "y", "origem", "conf det.", "dist. ATD"))
    tiles, points = [], []
    for frame in alarms:
        fused = best[frame][:2]
        atd = atd_pixel(ball, matrices, valid, frame)
        found = candidates.get(frame, [])
        conf, gap = nearest(found, fused)
        if atd is None:
            origin = "detector sozinho"
        elif np.hypot(atd[0] - fused[0], atd[1] - fused[1]) < 0.5:
            origin = "ATD"
        else:
            origin = "ATD refinado/trocado"
        conf_text = "-" if conf is None or gap > 1.0 else "{:.2f}".format(conf)
        atd_text = "-" if atd is None else "{:.0f} px".format(np.hypot(atd[0] - fused[0], atd[1] - fused[1]))
        print("{:>7} {:>8.1f} {:>8.1f}  {:<20} {:>9} {:>10}".format(frame, fused[0], fused[1], origin, conf_text, atd_text))
        points.append((frame, fused[0], fused[1]))

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            print("        (não consegui ler o frame {})".format(frame))
            continue
        marked = draw(image, fused, atd, found, args.top)
        cv2.imwrite(os.path.join(args.out_dir, "frame_{:06d}.jpg".format(frame)), marked)
        # à esquerda o recorte limpo (para enxergar a bola sem marcas por cima), à direita o marcado
        tile = np.hstack([crop_zoom(image, fused, args.half, args.zoom), crop_zoom(marked, fused, args.half, args.zoom)])
        label = "{}  {}  {}".format(frame, origin, "" if conf_text == "-" else "conf " + conf_text)
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.imwrite(os.path.join(args.out_dir, "crop_{:06d}.jpg".format(frame)), tile)
        tiles.append(tile)

    sheet = contact_sheet(tiles, args.columns)
    if sheet is not None:
        cv2.imwrite(os.path.join(args.out_dir, "contact_sheet.jpg"), sheet)

    print("\nPosições repetidas (a menos de {:g} px uma da outra):".format(args.cluster))
    repeated = [g for g in clusters(points, args.cluster) if len(g["frames"]) > 1]
    if not repeated:
        print("  nenhuma: os falsos alarmes estão espalhados (não parece um objeto fixo)")
    for group in repeated:
        print("  ({:.0f}, {:.0f}): {} frames -> {}".format(group["x"], group["y"], len(group["frames"]), group["frames"]))
    print("\nImagens em {} (frame_*.jpg inteiro, crop_*.jpg ampliado, contact_sheet.jpg).".format(args.out_dir))
    print("Vermelho = posição do tracker, azul = ATD, amarelo = candidatos do detector.")


if __name__ == "__main__":
    main()