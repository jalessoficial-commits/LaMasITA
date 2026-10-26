"""Exporta a bola do tracker final em coordenadas do campo, para TODOS os frames, como substituta da bola do ATD.

Colocar em tracking_bola/, junto de run_champion_full.py (usa a saída dele) e check_projection.py. Só numpy e
pandas:

    python tracking_bola/export_ball.py

Entrada: tracking_bola/out/full_bola.csv (do run_champion_full.py, em pixels, só frames avaliados com bola)
e o ATD bruto. Saída: <pasta do ATD>/BAR-ATH_tactical_ball_tracker.csv, ao lado do .txt do ATD, com UMA
linha por frame do vídeo (0 a 199.318):

    frame,x,y,fonte

  frame  índice do frame de vídeo, começando em 0 (o mesmo do arquivo de submissão; o frame_id do ATD é
         este + 1)
  x, y   posição da bola em coordenadas NORMALIZADAS do campo (0-1), as mesmas do ATD; vazio = sem bola.
         Para metros, multiplique por 105,3 x 68,0 (games.py). Fora das linhas pode passar um pouco de
         0-1 (a máscara aceita até 2 m fora).
  fonte  de onde veio a posição:
           detector          o ATD não tinha bola; posição do detector (Roboflow) sozinho
           detector_refinou  ATD e detector concordaram (até 40 px); vale a posição do detector
           detector_trocou   divergiram e o detector tinha conf >= 0,8; o detector passou por cima do ATD
           atd               divergiram sem confiança alta; posição original do ATD (sem arredondamento)
           sem_bola          frame avaliado em que o tracker não deu bola
           atd_fora_do_jogo  frame NÃO avaliado (antes do jogo, intervalo, depois): bola original do ATD

Posições do detector voltam ao campo pela inversa da homografia do frame. A homografia vale para o
plano do chão: bola alta aparece mais longe do que está (o ATD tem o mesmo problema).

POR QUE NA PASTA DO ATD: lá o arquivo fica fora do Git (data/ está no .gitignore). Ele é dado derivado do
vídeo, frame a frame, e NÃO pode ir para o repositório público (RULES.md). Para a equipe usar: cada um
regenera com os scripts, ou o arquivo é passado por fora do Git a quem também aceitou os termos.
"""

import argparse
import csv
import json
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

DEFAULT_TRACKER = os.path.join(HERE, "out", "full_bola.csv")
SOURCE = {"1 só detector": "detector", "2a concordam: detector": "detector_refinou",
          "2b divergem: detector confiante": "detector_trocou", "2c divergem: ATD": "atd"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="saída do run_champion_full.py (padrão out/full_bola.csv)")
    parser.add_argument("--out", default=None, help="padrão: <pasta do ATD>/<jogo>_tactical_ball_tracker.csv")
    parser.add_argument("--allow-partial", action="store_true", help="exporta mesmo se a detecção não cobriu o jogo todo")
    args = parser.parse_args(argv)

    atd_path = require(["atd_raw"])[0]
    out = args.out or os.path.join(os.path.dirname(atd_path), "{}_tactical_ball_tracker.csv".format(GAME))
    if not os.path.isfile(args.tracker):
        sys.exit("não achei {}. Rode o run_champion_full.py antes.".format(args.tracker))
    graded = np.zeros(source_frames("tactical", GAME), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    progress = os.path.join(os.path.dirname(args.tracker), "full_rf.csv.progress")
    if os.path.isfile(progress):
        with open(progress) as handle:
            done = int(json.load(handle)["done"])
        if done < int(graded.sum()) and not args.allow_partial:
            sys.exit("a detecção cobriu só {} de {} frames avaliados. Termine o run_champion_full.py "
                     "(ou use --allow-partial).".format(done, int(graded.sum())))

    n_frames = len(graded)
    ball = read_ball(atd_path, n_frames)
    matrices, _, valid = load_homography()
    fused = {}
    with open(args.tracker, newline="") as handle:
        for row in csv.DictReader(handle):
            fused[int(row["frame"])] = (float(row["x"]), float(row["y"]), row["regra"])

    rows, counts = [], {}
    for frame in range(n_frames):
        atd_xy = None if np.isnan(ball[frame, 0]) else (float(ball[frame, 0]), float(ball[frame, 1]))
        if not graded[frame]:
            x, y, source = (atd_xy or (None, None)) + ("atd_fora_do_jogo",)
        elif frame not in fused:
            x, y, source = None, None, "sem_bola"
        else:
            px, py, rule = fused[frame]
            source = SOURCE.get(rule)
            if source is None:
                sys.exit("regra desconhecida no frame {}: {}".format(frame, rule))
            if source == "atd":
                if atd_xy is None:
                    sys.exit("frame {}: a fusão usou o ATD, mas o ATD não tem bola".format(frame))
                x, y = atd_xy
            elif not valid[frame]:
                x, y, source = None, None, "sem_bola"
            else:
                u, v = project(np.linalg.inv(matrices[frame]), px, py)
                x, y = float(u), float(v)
        counts[source] = counts.get(source, 0) + 1
        rows.append((frame, x, y, source))

    # 1) comparação com o ATD, só nos frames avaliados (os que contam)
    tracker_has = np.array([r[1] is not None for r in rows])
    atd_has = ~np.isnan(ball[:n_frames, 0])
    print("BOLA NOS FRAMES AVALIADOS (os dois tempos)")
    print("  {:<12} {:>9} {:>14} {:>16}".format("trecho", "frames", "ATD com bola", "tracker c/ bola"))
    blocks = [("{}º tempo".format(half), first, last) for half, first, last in half_video_blocks(GAME)]
    for name, first, last in blocks + [("total", None, None)]:
        pick = graded.copy() if first is None else np.zeros(n_frames, dtype=bool)
        if first is not None:
            pick[first:last + 1] = True
        total = int(pick.sum())
        print("  {:<12} {:>9} {:>13.1f}% {:>15.1f}%".format(
            name, total, 100.0 * atd_has[pick].mean(), 100.0 * tracker_has[pick].mean()))
    both = int((tracker_has & atd_has & graded).sum())
    only_tracker = int((tracker_has & ~atd_has & graded).sum())
    only_atd = int((~tracker_has & atd_has & graded).sum())
    neither = int((~tracker_has & ~atd_has & graded).sum())
    total = int(graded.sum())
    print("\n  os dois têm bola          {:7d} ({:5.1f}%)".format(both, 100.0 * both / total))
    print("  só o tracker (recuperado) {:7d} ({:5.1f}%)".format(only_tracker, 100.0 * only_tracker / total))
    print("  só o ATD                  {:7d} ({:5.1f}%)".format(only_atd, 100.0 * only_atd / total))
    print("  nenhum                    {:7d} ({:5.1f}%)".format(neither, 100.0 * neither / total))
    print("\n  de onde veio a posição do tracker:")
    for source in ("detector", "detector_refinou", "detector_trocou", "atd", "sem_bola"):
        print("    {:<18} {:7d} ({:5.1f}%)".format(source, counts.get(source, 0), 100.0 * counts.get(source, 0) / total))

    # 2) grava o arquivo ao lado do ATD
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "fonte"])
        for frame, x, y, source in rows:
            writer.writerow([frame, "" if x is None else "{:.6f}".format(x), "" if y is None else "{:.6f}".format(y), source])
    print("\n{} frames (o vídeo inteiro) gravados em {}".format(n_frames, out))
    print("Fora dos tempos avaliados ({} frames) vale a bola original do ATD.".format(counts.get("atd_fora_do_jogo", 0)))
    print("ESTE ARQUIVO É DADO DERIVADO DO VÍDEO: não vai para o Git nem para o repositório público.")


if __name__ == "__main__":
    main()