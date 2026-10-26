"""O ATD "acha" a bola de verdade, ou repete a última posição / interpola entre detecções?

Colocar em tracking_bola/, junto de check_projection.py e eval_atd.py (este script importa deles). Só numpy:

    python tracking_bola/check_atd_nature.py
    python tracking_bola/check_atd_nature.py --tol 41

A documentação diz só que a bola ausente vira NaN e que o ATD é automático e sem correção. Não diz se
nos frames com bola a posição vem de uma detecção nova ou de uma estimativa (repetir a anterior,
ligar dois pontos em linha reta). Este script procura essas duas assinaturas nos dados:

  repetida  a posição é IDÊNTICA à do frame vizinho (anterior ou seguinte). Uma detecção nova numa bola
            em movimento quase nunca repete a posição até a última casa decimal.
  linear    a bola anda em linha reta com velocidade EXATAMENTE constante por 3 frames (segunda
            diferença nula em x e y). Interpolação linear entre dois pontos faz isso; uma detecção
            com ruído, quase nunca.
  nova      nenhuma das duas: parece uma medida de verdade.

Parte 1 conta essas classes no jogo todo (só o ATD, não usa suas anotações). Parte 2 cruza com as suas
anotações: o erro em pixels de cada classe. Se as classes "repetida" e "linear" têm erro muito maior,
as posições longes do ATD são estimativas velhas e não detecções erradas. A parte 3 mostra o erro pela
distância até o frame sem bola mais próximo: erro maior perto das lacunas sugere posições "esticadas".

Limite: a classificação é uma assinatura, não prova. Uma bola parada (cobrança de falta) repete a
posição de verdade, e uma bola rolando em linha reta pode parecer linear. Veja os números em conjunto.
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

DEFAULT_ANNOTATIONS = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
SAME = 1e-9          # posições "idênticas" (coordenadas 0-1)
LINEAR_EPS = 2e-5    # segunda diferença "nula" (coordenadas 0-1; o arquivo tem poucas casas decimais)
CLASSES = ("repetida", "linear", "nova")


def classify(ball):
    """Para cada frame, 'ausente', 'repetida', 'linear' ou 'nova' (array de texto)."""
    x, y = ball[:, 0], ball[:, 1]
    present = ~np.isnan(x)
    n = len(x)
    labels = np.full(n, "ausente", dtype=object)
    same_as_next = np.zeros(n, dtype=bool)          # o frame i e o i+1 têm a mesma posição
    both = present[:-1] & present[1:]
    same_as_next[:-1] = both & (np.abs(x[:-1] - x[1:]) < SAME) & (np.abs(y[:-1] - y[1:]) < SAME)
    repeated = same_as_next.copy()
    repeated[1:] |= same_as_next[:-1]
    linear = np.zeros(n, dtype=bool)
    triple = present[:-2] & present[1:-1] & present[2:]
    second_x = x[:-2] - 2 * x[1:-1] + x[2:]
    second_y = y[:-2] - 2 * y[1:-1] + y[2:]
    linear[1:-1] = triple & (np.abs(second_x) < LINEAR_EPS) & (np.abs(second_y) < LINEAR_EPS)
    for index in np.flatnonzero(present):
        labels[index] = "repetida" if repeated[index] else ("linear" if linear[index] else "nova")
    return labels


def distance_to_gap(present):
    """Para cada frame, quantos frames faltam até o frame sem bola mais próximo (n se não houver nenhum)."""
    n = len(present)
    big = n
    forward = np.full(n, big, dtype=np.int64)
    last = None
    for index in range(n):
        if not present[index]:
            last = index
        if last is not None:
            forward[index] = index - last
    backward = np.full(n, big, dtype=np.int64)
    nxt = None
    for index in range(n - 1, -1, -1):
        if not present[index]:
            nxt = index
        if nxt is not None:
            backward[index] = nxt - index
    return np.minimum(forward, backward)


def stats_line(errors, tol):
    errors = np.asarray(errors, dtype=float)
    if not len(errors):
        return "n = 0"
    return "n = {:4d}   acerto (até {:g} px) {:5.1f}%   erro mediano {:7.1f} px   erro médio {:7.1f} px".format(
        len(errors), tol, 100.0 * np.mean(errors <= tol), np.median(errors), np.mean(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--tol", type=float, default=41.0, help="tolerância de acerto em px (padrão 41)")
    args = parser.parse_args()

    n_frames = source_frames("tactical", GAME)
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    labels = classify(ball)
    present = labels != "ausente"

    print("PARTE 1: o ATD no jogo todo ({} frames)".format(n_frames))
    print("  com bola: {} ({:.1f}%)".format(int(present.sum()), 100.0 * present.mean()))
    for name in CLASSES:
        count = int((labels == name).sum())
        print("  {:<9} {:7d}  ({:.1f}% dos frames com bola)".format(name, count, 100.0 * count / max(1, present.sum())))

    if not os.path.isfile(args.annotations):
        print("\n(sem {}: partes 2 e 3 precisam das anotações)".format(args.annotations))
        return
    annotations = read_annotations(args.annotations)
    matrices, _, valid = load_homography()
    truth = {f: (v[1], v[2]) for f, v in annotations.items() if v[0] == "ball" and 0 <= f < n_frames}
    errors, gaps, classes = [], [], []
    to_gap = distance_to_gap(present)
    for frame, (cx, cy) in truth.items():
        pixel = atd_pixel(ball, matrices, valid, frame)
        if pixel is None:
            continue
        errors.append(float(np.hypot(pixel[0] - cx, pixel[1] - cy)))
        classes.append(labels[frame])
        gaps.append(int(to_gap[frame]))
    errors, gaps, classes = np.array(errors), np.array(gaps), np.array(classes)

    print("\nPARTE 2: erro do ATD por tipo de posição, nos seus {} frames anotados em que ele tem bola".format(len(errors)))
    print("  todos     " + stats_line(errors, args.tol))
    for name in CLASSES:
        print("  {:<9} {}".format(name, stats_line(errors[classes == name], args.tol)))

    print("\nPARTE 3: erro do ATD pela distância até o frame sem bola mais próximo")
    for low, high, label in ((1, 1, "colado na lacuna (1 frame)"), (2, 5, "2 a 5 frames"),
                             (6, 25, "6 a 25 frames (até 1 s)"), (26, n_frames, "mais de 25 frames")):
        pick = (gaps >= low) & (gaps <= high)
        print("  {:<28} {}".format(label, stats_line(errors[pick], args.tol)))

    print("\nComo ler: se 'repetida' e 'linear' têm muito mais erro que 'nova', o ATD está devolvendo")
    print("estimativas (posição velha ou ligada em linha reta), não detecções, e o erro dele é de")
    print("'bola andou desde a última detecção'. Se as três classes têm erro parecido, o erro vem das")
    print("próprias detecções (calibração, ponto de contato, confusão com outro objeto).")


if __name__ == "__main__":
    main()