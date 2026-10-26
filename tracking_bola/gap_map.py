"""Mapa de lacunas da bola no ATD (etapa 0.2).

Colocar em tracking_bola/ (um nível abaixo da raiz, onde está o games.py). Só numpy.

    python tracking_bola/gap_map.py

Só nos frames avaliados (os dois tempos), faz três coisas:
  1. Distribui as lacunas de bola por duração.
  2. Lista as maiores lacunas e diz se cada uma toca uma bola parada marcada no Smart Tagging.
  3. Nos frames sem bola da janela pública, mostra quantos são D e quantos são H/A no GT público.

Serve para decidir se o tempo sem bola é jogo vivo (o detector se justifica) ou parada
(o rótulo certo já é D e a posição da bola importa pouco).

Checagem de sanidade: DATA.md diz 94.214 frames avaliados com bola, de 158.857.
O Smart Tagging é automático e suas janelas são aproximadas; por isso há tolerância.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402

FPS = 25
TOLERANCE = 75   # frames (3 s) de folga ao comparar lacunas com as janelas do Smart Tagging
TOP = 20         # quantas lacunas listar
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
# Faixas em frames: (0, 0,2 s], (0,2, 1 s], (1, 3 s], > 3 s
BINS = (("<= 0,2 s", 1, 5), ("0,2-1 s", 6, 25), ("1-3 s", 26, 75), ("> 3 s", 76, 10 ** 9))


def ball_presence(raw_path, n_frames):
    """Vetor booleano por frame de vídeo: True onde o ATD tem a bola."""
    seen = np.zeros(n_frames, dtype=bool)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            if parts[2][0] != "N":  # "NaN" = bola ausente
                seen[int(parts[0]) - 1] = True  # frame_id conta de 1, vídeo de 0
    return seen


def gaps_of(visible, offset):
    """(frame inicial, comprimento) de cada corrida de False. `offset` converte para frame de vídeo."""
    padded = np.concatenate(([True], visible, [True])).astype(np.int8)
    change = np.diff(padded)
    starts = np.flatnonzero(change == -1)
    ends = np.flatnonzero(change == 1)
    return starts + offset, ends - starts


def nearby_piece(start, end, pieces):
    """(índice da bola parada mais próxima do início da lacuna, quantas tocam a lacuna).

    "Toca" = a janela da bola parada, com folga de TOLERANCE, cruza a lacuna [start, end].
    Sem nenhuma, devolve (-1, 0). A mais próxima do início é a candidata a ter causado a lacuna.
    """
    touching = [index for index, (p_start, p_end, _) in enumerate(pieces)
                if p_start <= end + TOLERANCE and p_end >= start - TOLERANCE]
    if not touching:
        return -1, 0
    best = min(touching, key=lambda index: abs(pieces[index][0] - start))
    return best, len(touching)


def read_public_gt(path, n_frames):
    """Rótulos do GT público como array de strings, um por frame de vídeo."""
    labels = []
    with open(path) as handle:
        next(handle)  # cabeçalho frame,label
        for line in handle:
            labels.append(line.strip().split(",")[1])
    if len(labels) != n_frames:
        raise ValueError("GT público com {} linhas, esperado {}".format(len(labels), n_frames))
    return np.array(labels)


if __name__ == "__main__":
    n_frames = source_frames("tactical", GAME)
    seen = ball_presence(require(["atd_raw"])[0], n_frames)

    all_starts, all_lengths = [], []
    graded = present = 0
    for _, first, last in half_video_blocks(GAME):
        segment = seen[first:last + 1]
        graded += len(segment)
        present += int(segment.sum())
        starts, lengths = gaps_of(segment, first)
        all_starts.append(starts)
        all_lengths.append(lengths)
    starts = np.concatenate(all_starts)
    lengths = np.concatenate(all_lengths)
    ends = starts + lengths - 1

    # --- 1. distribuição das lacunas ------------------------------------------------------
    print("frames avaliados: {}  com bola: {} ({:.1f}%)".format(
        graded, present, 100.0 * present / graded))
    print("lacunas: {}  frames sem bola: {}\n".format(len(lengths), int(lengths.sum())))
    print("{:<10} {:>8} {:>12} {:>14}".format("faixa", "lacunas", "frames", "% sem bola"))
    for name, low, high in BINS:
        inside = lengths[(lengths >= low) & (lengths <= high)]
        print("{:<10} {:>8} {:>12} {:>13.1f}%".format(
            name, len(inside), int(inside.sum()), 100.0 * inside.sum() / lengths.sum()))
    print("\nmediana {:.0f} frames, p90 {:.0f}, máx {}".format(
        np.median(lengths), np.percentile(lengths, 90), int(lengths.max())))

    # --- 2. cruzamento com o Smart Tagging ------------------------------------------------
    pieces = [(event["start_frame"], event["end_frame"], event["tags"].get("Type") or "?")
              for event in load_tags()
              if (event["code"] or "").strip().lower() == "set pieces"]
    matches = [nearby_piece(s, e, pieces) for s, e in zip(starts, ends)]
    hit = np.array([m[0] for m in matches])
    n_touch = np.array([m[1] for m in matches])
    near = hit >= 0

    print("\n--- Smart Tagging: {} bolas paradas marcadas, folga de {} s ---".format(
        len(pieces), TOLERANCE // FPS))
    print("lacunas que tocam uma bola parada: {} de {}  ({:.1f}% dos frames sem bola)".format(
        int(near.sum()), len(lengths), 100.0 * lengths[near].sum() / lengths.sum()))
    print("lacunas sem nenhuma bola parada por perto: {}  ({:.1f}% dos frames sem bola)".format(
        int((~near).sum()), 100.0 * lengths[~near].sum() / lengths.sum()))

    print("\n{} maiores lacunas".format(TOP))
    print("{:>3} {:>8} {:>8} {:>8}  {}".format(
        "#", "início", "fim", "dur (s)", "bola parada por perto (janela no Smart Tagging)"))
    for rank, index in enumerate(np.argsort(-lengths)[:TOP], start=1):
        if near[index]:
            p_start, p_end, p_type = pieces[hit[index]]
            piece = "{} ({:.1f} s)".format(p_type, (p_end - p_start) / float(FPS))
            if n_touch[index] > 1:
                piece += " +{} mais".format(n_touch[index] - 1)
        else:
            piece = "-"
        print("{:>3} {:>8} {:>8} {:>8.1f}  {}".format(
            rank, int(starts[index]), int(ends[index]), lengths[index] / float(FPS), piece))

    print("\nDuração da janela de cada tipo de bola parada no Smart Tagging (s)")
    print("{:<24} {:>4} {:>7} {:>7} {:>7}".format("tipo", "n", "média", "mín", "máx"))
    for name in sorted({p[2] for p in pieces}):
        durations = np.array([(e - s) / float(FPS) for s, e, t in pieces if t == name])
        print("{:<24} {:>4} {:>7.1f} {:>7.1f} {:>7.1f}".format(
            name, len(durations), durations.mean(), durations.min(), durations.max()))
    print("Atenção: é a janela marcada em volta do reinício (cerca de 5 s), não a duração real")
    print("da bola morta. Para a duração real, use as corridas de D do GT (só na janela pública).")

    # --- 3. cruzamento com o GT público ---------------------------------------------------
    labels = read_public_gt(GT_PUBLIC, n_frames)
    inside_gt = labels != "X"

    def count(mask):
        return [int(np.sum(labels[mask] == label)) for label in ("H", "A", "D")]

    def pct(part, whole):
        return 100.0 * part / whole if whole else 0.0

    without_ball = count(inside_gt & ~seen)
    with_ball = count(inside_gt & seen)
    total = [a + b for a, b in zip(without_ball, with_ball)]
    n_without, n_with, n_total = sum(without_ball), sum(with_ball), sum(total)
    row = "{:<22} {:>6} {:>6} {:>6} {:>8}"

    print("\n--- Comparação com o GT público ---")
    print("Pergunta: nos frames em que o ATD NÃO tem a bola, qual é o rótulo verdadeiro?")
    print("Só vale na janela pública ({} frames, os primeiros 5 min de jogo).\n".format(n_total))
    print(row.format("", "H", "A", "D", "total"))
    print(row.format("ATD sem bola", *(without_ball + [n_without])))
    print(row.format("ATD com bola", *(with_ball + [n_with])))
    print(row.format("total (GT)", *(total + [n_total])))
    print("H = posse do mandante, A = posse do visitante, D = bola morta\n")

    print("O que a tabela diz")
    print("1. {} frames da janela estão sem bola ({:.1f}%). Desses:".format(
        n_without, pct(n_without, n_total)))
    print("   - {:.1f}% são bola morta (D): o rótulo certo já é D, a posição da bola não faz falta."
          .format(pct(without_ball[2], n_without)))
    print("   - {:.1f}% são jogo vivo (H ou A): aqui a posição ajudaria, e é onde um detector pode ganhar."
          .format(pct(without_ball[0] + without_ball[1], n_without)))
    print("2. Dos {} frames de bola morta (D), {:.1f}% estão sem bola no ATD e {:.1f}% têm bola.".format(
        total[2], pct(without_ball[2], total[2]), pct(with_ball[2], total[2])))
    print("   Quanto menor o primeiro número, mais fraco é 'sem bola' como sinal de D.")
    print("\nCuidado: a janela tem {:.1f}% de frames sem bola, contra {:.1f}% no total avaliado, e só 17"
          .format(pct(n_without, n_total), pct(graded - present, graded)))
    print("posses. Serve para ver o padrão, não para extrapolar as proporções para a partida toda.")