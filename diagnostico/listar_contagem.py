"""Lista os trechos (frames graduados) em que um time tem slots demais ou de menos.

    python diagnostico/listar_contagem.py casa --min 12      # casa com 12 ou mais slots
    python diagnostico/listar_contagem.py fora --min 12
    python diagnostico/listar_contagem.py casa --max 2       # casa com 2 ou menos slots

Imprime os trechos mais longos com os slots presentes no início de cada um. O início de um trecho
vai direto no assistir_slot.py com --exato. Usa o cache do diagnostico.py.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))  # baseline, load_homography
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import baseline  # noqa: E402

# Params
CACHE = os.path.join(DATA_DIR, "_cache", "atd_xy_v2.npz")
FPS = GAMES[GAME]["fps"]
TOP = 20

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("time", choices=("casa", "fora"))
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--min", type=int, help="frames com pelo menos N slots do time")
group.add_argument("--max", type=int, help="frames com no máximo N slots do time")
args = parser.parse_args()

xy = np.load(CACHE)["xy"]
team = baseline.team_of_each_track(asset_paths()["atd_meta"])
side = 0 if args.time == "casa" else 1
present = ~np.isnan(xy[:, :, 0])
count = present[:, team == side].sum(axis=1)

graded = np.zeros(len(xy), dtype=bool)
for _, first, last in half_video_blocks(GAME):
    graded[first:last + 1] = True
hit = graded & ((count >= args.min) if args.min is not None else (count <= args.max))

padded = np.concatenate(([0], hit.astype(np.int8), [0]))
starts = np.flatnonzero(np.diff(padded) == 1)
ends = np.flatnonzero(np.diff(padded) == -1)
order = np.argsort(-(ends - starts))[:TOP]

print("%s %s: %d frames graduados em %d trechos" % (
    args.time, ("com >= %d slots" % args.min) if args.min is not None else
    ("com <= %d slots" % args.max), hit.sum(), len(starts)))
print("\n%d trechos mais longos:" % len(order))
print("%-8s %-8s %8s  %s" % ("início", "fim", "duração", "slots do time presentes no início"))
for i in order:
    a, b = starts[i], ends[i]
    slots = np.flatnonzero(present[a] & (team == side)) + 1
    print("%-8d %-8d %7.1fs  %s" % (a, b - 1, (b - a) / FPS, " ".join(map(str, slots))))