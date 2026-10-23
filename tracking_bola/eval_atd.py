"""Mede quanto o ATD erra a bola, usando as suas anotações à mão como gabarito.

Colocar em tracking_bola/, junto de check_projection.py (este script importa dele duas funções).
Só numpy:

    python tracking_bola/eval_atd.py
    python tracking_bola/eval_atd.py --annotations tracking_bola/out/annotations/ball_annotations.csv

Entrada: ball_annotations.csv (da annotate_ball.py). A bola do ATD, em coordenadas do campo (0-1),
é projetada em pixels do vídeo com a homografia do próprio frame e comparada com o seu clique.
Frames com status "skip" ficam de fora. "uncertain" é medido à parte, nunca misturado com "ball".

O que o relatório responde:
  1. Concordância: quando você vê a bola, o ATD tem bola? Quando você NÃO vê, o ATD diz ter?
  2. Erro de posição (em pixels) nos frames em que os dois têm bola, com quantos acertam por faixa.
  3. Viés: o ATD fica sistematicamente para um lado (por exemplo, abaixo da bola)?
  4. Rápido vs lento: o erro é maior quando a bola está rápida (provável bola alta)?
  5. Deslocamento no tempo: o ATD do frame f+k bate melhor com o seu clique do frame f?
  6. Os piores frames, para você abrir com check_projection.py --frames.

Cuidados de leitura:
  - Erro em PIXELS, não em metros: o mesmo erro em pixels vale mais ou menos metros conforme a
    posição no campo. Compare com o tamanho da bola que você viu na ferramenta de anotação.
  - "ATD diz ter bola e você não vê" não prova erro do ATD: ele pode estimar uma bola que está
    escondida atrás de um jogador. É um alerta, não uma sentença.
  - Com poucos frames as porcentagens têm margem larga; o intervalo (95%) está ao lado de cada uma.
  - A coluna de velocidade do ATD (3º número da bola) é assumida, não confirmada na documentação.

Saída detalhada por frame: tracking_bola/out/annotations/atd_error.csv (fora do git).
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
from load_homography import load as load_homography, project  # noqa: E402
from check_projection import read_ball  # noqa: E402

DEFAULT_IN = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
DEFAULT_OUT = os.path.join(HERE, "out", "annotations", "atd_error.csv")
THRESHOLDS = (10, 25, 50)   # faixas de erro em pixels
MAX_SHIFT = 10              # testa deslocamentos de -MAX_SHIFT a +MAX_SHIFT frames
MIN_GROUP = 5               # menos frames que isso: não resume
WORST = 10


def read_annotations(path):
    """{frame: (status, x, y)}; se um frame aparecer duas vezes, vale a última linha."""
    result = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            x = float(row["x"]) if row["x"] else np.nan
            y = float(row["y"]) if row["y"] else np.nan
            result[int(row["frame"])] = (row["status"], x, y)
    return result


def wilson(hits, n, z=1.96):
    """Intervalo de Wilson (95%) para uma proporção, em porcentagem."""
    if n == 0:
        return 0.0, 0.0
    p = hits / float(n)
    denominator = 1 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / denominator
    return 100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half)


def share(hits, n):
    low, high = wilson(hits, n)
    return "{:.0f}% ({} de {}; intervalo {:.0f}-{:.0f}%)".format(
        100.0 * hits / n if n else 0.0, hits, n, low, high)


def atd_pixel(ball, matrices, valid, frame, shift=0):
    """Pixel onde a bola do ATD do frame (frame+shift) aparece na imagem do frame `frame`.
    None se o ATD não tem bola nesse frame ou não há homografia válida."""
    other = frame + shift
    if not (0 <= frame < len(matrices)) or not (0 <= other < len(ball)):
        return None
    if np.isnan(ball[other, 0]) or not valid[frame]:
        return None
    x, y = project(matrices[frame], ball[other, 0], ball[other, 1])
    return float(x), float(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", default=DEFAULT_IN)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)

    n_frames = source_frames("tactical", GAME)
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, _, valid = load_homography()

    counts = {}
    for status, _, _ in annotations.values():
        counts[status] = counts.get(status, 0) + 1
    print("anotações: " + ", ".join("{} {}".format(k, counts[k]) for k in sorted(counts)))
    print("(skip fica de fora de tudo abaixo)\n")

    # --- 1. concordância ------------------------------------------------------------------
    atd_has = {}
    for frame, (status, _, _) in annotations.items():
        if status != "skip" and 0 <= frame < n_frames:
            atd_has[frame] = not np.isnan(ball[frame, 0])
    rows = []
    for status in ("ball", "uncertain", "not_visible"):
        frames = [f for f, (s, _, _) in annotations.items() if s == status and f in atd_has]
        yes = sum(atd_has[f] for f in frames)
        rows.append((status, yes, len(frames) - yes))
    label = {"ball": "você viu a bola", "uncertain": "você ficou em dúvida", "not_visible": "você não viu a bola"}
    print("1. CONCORDÂNCIA (o ATD tem bola neste frame?)")
    print("{:<24} {:>14} {:>14}".format("", "ATD com bola", "ATD sem bola"))
    for status, yes, no in rows:
        print("{:<24} {:>14} {:>14}".format(label[status], yes, no))
    ball_row, hidden_row = rows[0], rows[2]
    if ball_row[1] + ball_row[2]:
        print("  -> Onde você vê a bola, o ATD NÃO tem bola em {}.".format(
            share(ball_row[2], ball_row[1] + ball_row[2])))
    if hidden_row[1] + hidden_row[2]:
        print("  -> Onde você NÃO vê a bola, o ATD diz ter bola em {}.".format(
            share(hidden_row[1], hidden_row[1] + hidden_row[2])))
        print("     (pode ser bola escondida que o ATD estima, ou falso positivo: não dá para distinguir.)")
    print()

    # --- erro de posição ------------------------------------------------------------------
    records = []
    for frame in sorted(annotations):
        status, ax, ay = annotations[frame]
        if status not in ("ball", "uncertain") or not (0 <= frame < n_frames):
            continue
        pixel = atd_pixel(ball, matrices, valid, frame)
        if pixel is None:
            continue
        records.append({"frame": frame, "status": status, "ann_x": ax, "ann_y": ay,
                        "atd_x": pixel[0], "atd_y": pixel[1],
                        "dist": float(np.hypot(pixel[0] - ax, pixel[1] - ay)),
                        "speed": float(ball[frame, 2])})
    sure = [r for r in records if r["status"] == "ball"]
    unsure = [r for r in records if r["status"] == "uncertain"]

    print("2. ERRO DE POSIÇÃO (pixels; só frames em que você viu a bola E o ATD tem bola)")
    if len(sure) < MIN_GROUP:
        print("  poucos frames ({}); anote mais antes de tirar conclusões.\n".format(len(sure)))
    else:
        dist = np.array([r["dist"] for r in sure])
        print("  n = {}   mediana {:.1f}   p75 {:.1f}   p90 {:.1f}   máximo {:.1f}".format(
            len(dist), np.median(dist), np.percentile(dist, 75), np.percentile(dist, 90), dist.max()))
        for limit in THRESHOLDS:
            print("  erro até {:>3} px: {}".format(limit, share(int((dist <= limit).sum()), len(dist))))
        print("  Compare esses pixels com o tamanho da bola na ferramenta de anotação: um erro de")
        print("  alguns diâmetros de bola já é um rótulo de treino errado.")
        if unsure:
            print("  (frames 'uncertain', à parte: n = {}, mediana {:.1f} px)".format(
                len(unsure), np.median([r["dist"] for r in unsure])))
        print()

        # --- 3. viés ---------------------------------------------------------------------
        dx = np.median([r["atd_x"] - r["ann_x"] for r in sure])
        dy = np.median([r["atd_y"] - r["ann_y"] for r in sure])
        print("3. VIÉS (mediana de ATD menos seu clique; y cresce para BAIXO na imagem)")
        print("  dx = {:+.1f} px   dy = {:+.1f} px".format(dx, dy))
        print("  Perto de zero: sem viés. dy positivo e consistente = o ATD fica ABAIXO da bola,")
        print("  o que combina com o ATD guardar o ponto do chão sob a bola no ar.\n")

        # --- 4. rápido vs lento ----------------------------------------------------------
        speeds = np.array([r["speed"] for r in sure])
        print("4. RÁPIDO VS LENTO (pela velocidade que o ATD reporta para a bola; coluna assumida)")
        if np.isfinite(speeds).sum() >= 2 * MIN_GROUP:
            cut = float(np.nanmedian(speeds))
            fast = [r["dist"] for r in sure if r["speed"] > cut]
            slow = [r["dist"] for r in sure if r["speed"] <= cut]
            if len(fast) >= MIN_GROUP and len(slow) >= MIN_GROUP:
                print("  lenta (<= {:.1f}): n = {}, mediana {:.1f} px".format(cut, len(slow), np.median(slow)))
                print("  rápida (> {:.1f}): n = {}, mediana {:.1f} px".format(cut, len(fast), np.median(fast)))
                print("  Erro bem maior na rápida sugere bola alta ou atraso do ATD.\n")
            else:
                print("  grupos pequenos demais.\n")
        else:
            print("  sem velocidade suficiente nos dados.\n")

        # --- 5. deslocamento no tempo ----------------------------------------------------
        print("5. DESLOCAMENTO NO TEMPO (seu clique do frame f contra o ATD do frame f+k)")
        print("  {:>4} {:>6} {:>14}".format("k", "n", "mediana (px)"))
        medians = {}
        for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
            errors = []
            for r in sure:
                pixel = atd_pixel(ball, matrices, valid, r["frame"], shift)
                if pixel is not None:
                    errors.append(np.hypot(pixel[0] - r["ann_x"], pixel[1] - r["ann_y"]))
            if len(errors) >= MIN_GROUP:
                medians[shift] = float(np.median(errors))
                print("  {:>+4d} {:>6} {:>14.1f}".format(shift, len(errors), medians[shift]))
        if 0 in medians and len(medians) >= 3:
            best = min(medians, key=medians.get)
            if best != 0 and medians[best] < 0.7 * medians[0]:
                print("  -> O melhor k é {:+d} ({:.1f} px contra {:.1f} em k = 0). O ATD no frame f{:+d} coincide".format(
                    best, medians[best], medians[0], best))
                print("     com o vídeo no frame f: há um deslocamento de {} frames entre ATD e vídeo".format(abs(best)))
                print("     (ou o seek do vídeo está deslocado). Resolva isto antes de treinar.\n")
            else:
                print("  -> Sem evidência de deslocamento: k = 0 é o melhor ou está empatado.\n")
        else:
            print()

        # --- 6. piores frames ------------------------------------------------------------
        worst = sorted(sure, key=lambda r: -r["dist"])[:WORST]
        print("6. {} PIORES FRAMES (abra com: python tracking_bola/check_projection.py --frames ...)".format(len(worst)))
        print("  " + "  ".join("{}({:.0f}px)".format(r["frame"], r["dist"]) for r in worst))
        print("  Nota: o check_projection mostra o frame que você anotou, com o ATD projetado.\n")

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "status", "ann_x", "ann_y", "atd_x", "atd_y", "dist", "speed"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("{:.2f}".format(v) if isinstance(v, float) else v) for k, v in r.items()})
    print("detalhe por frame: {}".format(args.out))


if __name__ == "__main__":
    main()