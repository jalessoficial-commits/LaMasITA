"""Vídeo curto comparando a bola do ATD com a do tracker novo, para mostrar à equipe.

Colocar em tracking_bola/, junto de check_projection.py. Precisa de opencv-python. Usa o arquivo do
export_ball.py (data/tactical/BAR-ATH_tactical_ball_tracker.csv):

    python tracking_bola/make_comparison_video.py                         # 3 trechos de 30 s, escolhidos sozinho
    python tracking_bola/make_comparison_video.py --clips 1 --clip-seconds 90 --start 45000

Desenho em cada frame:
  azul      bola do ATD (projetada com a homografia)
  vermelho  bola do tracker novo
  barra     no topo: tempo de jogo e o que cada um diz ("bola" / "sem bola"), com a origem da posição do tracker
  lupa      canto superior direito: recorte ampliado em volta da bola, para enxergar a bola de ~8 px

Escolha dos trechos (sem --start): os --clips trechos de --clip-seconds, sem sobreposição e dentro dos
tempos avaliados, com mais RECUPERAÇÕES EM JOGO VIVO: frames em que só o tracker tem bola, dentro de uma
lacuna CURTA do ATD (até --max-gap frames, padrão 50 = 2 s) que tem bola do ATD dos dois lados. Lacunas
longas costumam ser jogo parado (bola fora, falta, comemoração) e ficam de fora, assim como o trecho antes
do pontapé do 2º tempo (frames 110.808-113.741, bola parada; ver FAQ). Com --start, um trecho começando
nesse frame. Cada trecho abre com 1,5 s de legenda.

Saída: tracking_bola/out/video_comparacao.mp4 (padrão 1280x720; --width muda).
É UM RECORTE DO VÍDEO DA COMPETIÇÃO: mostre à equipe, mas não publique nem suba para o Git.
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
from check_projection import read_ball  # noqa: E402

FPS = 25
SECOND_HALF_KICKOFF = 113742    # FAQ: o 2º tempo é avaliado desde 110.808, mas a bola só rola a partir daqui
BLUE, RED, WHITE, BLACK, YELLOW = (255, 140, 0), (40, 40, 255), (255, 255, 255), (0, 0, 0), (0, 230, 255)
LABELS = {"detector": "detector (ATD sem bola)", "detector_refinou": "detector refinou o ATD",
          "detector_trocou": "detector corrigiu o ATD", "atd": "manteve o ATD", "sem_bola": "-"}


def read_tracker(path, n_frames):
    """(n, 2) com x, y normalizados (NaN = sem bola) e lista de fontes, do CSV do export_ball.py."""
    xy = np.full((n_frames, 2), np.nan)
    source = [""] * n_frames
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if 0 <= frame < n_frames:
                source[frame] = row["fonte"]
                if row["x"] != "":
                    xy[frame] = (float(row["x"]), float(row["y"]))
    return xy, source


def pick_windows(score, blocks, length, count):
    """Até `count` janelas [início, fim) de `length` frames, sem sobreposição, dentro dos blocos, com maior soma de score."""
    candidates = []
    for _, first, last in blocks:
        if last - first + 1 < length:
            continue
        cumulative = np.concatenate([[0], np.cumsum(score[first:last + 1])])
        sums = cumulative[length:] - cumulative[:-length]
        for offset in np.argsort(-sums, kind="stable"):
            candidates.append((float(sums[offset]), first + int(offset)))
    candidates.sort(key=lambda c: -c[0])
    chosen = []
    for value, start in candidates:
        if all(abs(start - other) >= length for _, other in chosen):
            chosen.append((value, start))
        if len(chosen) == count:
            break
    return sorted(start for _, start in chosen)


def live_recoveries(only_tracker, atd_has, max_gap):
    """Marca os frames 'só o tracker' que estão em lacunas do ATD de até max_gap frames com bola do ATD
    nos dois lados (o ATD piscou em jogo vivo)."""
    score = np.zeros(len(atd_has))
    frame, n = 0, len(atd_has)
    while frame < n:
        if atd_has[frame]:
            frame += 1
            continue
        end = frame
        while end < n and not atd_has[end]:
            end += 1
        bounded = frame > 0 and end < n
        if bounded and end - frame <= max_gap:
            score[frame:end] = only_tracker[frame:end]
        frame = end
    return score


def to_pixel(matrix, xy):
    if np.isnan(xy[0]) or not np.all(np.isfinite(matrix)):
        return None
    x, y = project(matrix, xy[0], xy[1])
    return float(x), float(y)


def game_clock(frame, blocks):
    for half, first, last in blocks:
        if first <= frame <= last:
            seconds = (frame - first) // FPS
            return "{}º tempo  {:02d}:{:02d}".format(half, seconds // 60, seconds % 60)
    return "fora do jogo"


def draw_frame(cv2, image, atd, tracker, source, clock, focus, inset, scale_out, counts):
    marked = image.copy()
    if atd is not None:
        cv2.circle(marked, (int(round(atd[0])), int(round(atd[1]))), 16, BLUE, 3)
    if tracker is not None:
        cv2.circle(marked, (int(round(tracker[0])), int(round(tracker[1]))), 10, RED, 3)

    height, width = marked.shape[:2]
    out = cv2.resize(marked, (int(width * scale_out), int(height * scale_out)), interpolation=cv2.INTER_AREA)
    if focus is not None:
        half = inset // 2
        cx = int(min(max(focus[0], half), width - half))
        cy = int(min(max(focus[1], half), height - half))
        crop = marked[cy - half:cy + half, cx - half:cx + half]
        zoom = cv2.resize(crop, (inset * 3, inset * 3), interpolation=cv2.INTER_NEAREST)
        size = min(zoom.shape[0], out.shape[0] // 2)
        zoom = cv2.resize(zoom, (size, size), interpolation=cv2.INTER_NEAREST)
        out[50:50 + size, out.shape[1] - size - 10:out.shape[1] - 10] = zoom
        cv2.rectangle(out, (out.shape[1] - size - 10, 50), (out.shape[1] - 10, 50 + size), WHITE, 2)

    cv2.rectangle(out, (0, 0), (out.shape[1], 42), BLACK, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, clock, (10, 29), font, 0.7, WHITE, 2)
    cv2.putText(out, "ATD: " + ("bola" if atd is not None else "sem bola"), (250, 29), font, 0.7, BLUE, 2)
    tracker_text = "Tracker: " + ("bola - " + LABELS.get(source, source) if tracker is not None else "sem bola")
    cv2.putText(out, tracker_text, (470, 29), font, 0.7, RED, 2)
    footer = "neste trecho: so o tracker tem bola em {} frames; so o ATD em {}".format(counts[0], counts[1])
    cv2.rectangle(out, (0, out.shape[0] - 34), (out.shape[1], out.shape[0]), BLACK, -1)
    cv2.putText(out, footer, (10, out.shape[0] - 11), font, 0.6, WHITE, 1)
    return out


def title_card(cv2, frame_image, lines):
    out = (frame_image * 0.35).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, (text, color) in enumerate(lines):
        size = cv2.getTextSize(text, font, 1.0, 2)[0]
        x = (out.shape[1] - size[0]) // 2
        y = out.shape[0] // 2 - 40 + 45 * index
        cv2.putText(out, text, (x, y), font, 1.0, color, 2)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", default=None, help="CSV do export_ball.py (padrão: o da pasta do ATD)")
    parser.add_argument("--clips", type=int, default=3, help="quantos trechos (padrão 3)")
    parser.add_argument("--clip-seconds", type=float, default=30.0, help="duração de cada trecho (padrão 30 s)")
    parser.add_argument("--start", type=int, default=None, help="frame inicial de um trecho escolhido por você")
    parser.add_argument("--max-gap", type=int, default=50, help="maior lacuna do ATD que conta como jogo vivo, em frames (padrão 50)")
    parser.add_argument("--width", type=int, default=1280, help="largura do vídeo de saída (padrão 1280)")
    parser.add_argument("--inset", type=int, default=120, help="lado do recorte da lupa, em px do vídeo (padrão 120)")
    parser.add_argument("--out", default=os.path.join(HERE, "out", "video_comparacao.mp4"))
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  python -m pip install opencv-python")

    atd_path = require(["atd_raw"])[0]
    tracker_path = args.tracker or os.path.join(os.path.dirname(atd_path), "{}_tactical_ball_tracker.csv".format(GAME))
    if not os.path.isfile(tracker_path):
        sys.exit("não achei {}. Rode o export_ball.py antes.".format(tracker_path))
    n_frames = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)
    atd = read_ball(atd_path, n_frames)[:, :2]
    tracker, sources = read_tracker(tracker_path, n_frames)
    matrices, _, valid = load_homography()
    length = max(1, int(round(args.clip_seconds * FPS)))

    only_tracker = (~np.isnan(tracker[:, 0])) & np.isnan(atd[:, 0])
    if args.start is not None:
        starts = [args.start]
    else:
        score = live_recoveries(only_tracker, ~np.isnan(atd[:, 0]), args.max_gap)
        score[110808:SECOND_HALF_KICKOFF] = 0       # antes do pontapé do 2º tempo: bola parada
        starts = pick_windows(score, blocks, length, args.clips)
    if not starts:
        sys.exit("nenhum trecho cabe nos tempos avaliados com essa duração")

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_out = args.width / float(width)
    size_out = (int(width * scale_out), int(height * scale_out))
    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size_out)

    for number, start in enumerate(starts, start=1):
        end = min(start + length, n_frames)
        counts = (int(only_tracker[start:end].sum()),
                  int(((~np.isnan(atd[start:end, 0])) & np.isnan(tracker[start:end, 0])).sum()))
        print("trecho {}: frames {}-{} ({}), só o tracker tem bola em {} frames, só o ATD em {}".format(
            number, start, end - 1, game_clock(start, blocks), counts[0], counts[1]))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        focus = None
        for frame in range(start, end):
            ok, image = capture.read()
            if not ok:
                print("  não consegui ler o frame {}; trecho encerrado".format(frame))
                break
            matrix = matrices[frame] if valid[frame] else np.full((3, 3), np.nan)
            atd_px = to_pixel(matrix, atd[frame])
            tracker_px = to_pixel(matrix, tracker[frame])
            focus = tracker_px or atd_px or focus
            out = draw_frame(cv2, image, atd_px, tracker_px, sources[frame], game_clock(frame, blocks),
                             focus, args.inset, scale_out, counts)
            if frame == start:
                card = title_card(cv2, out, [
                    ("Trecho {} de {}  -  {}".format(number, len(starts), game_clock(start, blocks)), WHITE),
                    ("azul = ATD (tracker antigo)", BLUE),
                    ("vermelho = tracker novo", RED),
                    ("lupa no canto: bola ampliada", YELLOW)])
                for _ in range(int(1.5 * FPS)):
                    writer.write(card)
            writer.write(out)
    writer.release()
    print("vídeo: {} ({}x{}, {} fps)".format(args.out, size_out[0], size_out[1], FPS))
    print("É um recorte do vídeo da competição: mostre à equipe, mas não publique nem suba para o Git.")


if __name__ == "__main__":
    main()