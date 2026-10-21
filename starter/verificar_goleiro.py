"""Quais slots se comportam como goleiro? Sem vídeo, só pelo comportamento no tracking.

    python starter/verificar_goleiros.py

Para cada slot, nos frames graduados em que está presente:
  - % do tempo em que é o jogador MAIS RECUADO do próprio time (mais perto do próprio gol)
  - % do tempo dentro da própria grande área
  - distância mediana até a própria linha de fundo, em metros

Um goleiro fica perto de 100% na primeira coluna. Que lado cada time defende em cada tempo é
decidido no frame do kickoff de cada tempo, quando todos estão no próprio campo -- mais robusto
que a média do tempo inteiro, que ficou perto demais de 0.5 no diagnóstico.

Usa o cache do tracking_diagnostics.py.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
import baseline  # noqa: E402

# Params
CACHE = os.path.join(DATA_DIR, "_cache", "atd_xy_v2.npz")
LENGTH, WIDTH = GAMES[GAME]["pitch"]
KICKOFFS = {1: 11185, 2: 113742}         # frames de kickoff publicados em DATA.md / FAQ.md
AREA_DEPTH, AREA_WIDTH = 16.5, 40.3      # grande área, metros
MIN_FRAMES = 1000

xy = np.load(CACHE)["xy"]
team = baseline.team_of_each_track(asset_paths()["atd_meta"])
present_all = ~np.isnan(xy[:, :, 0])

rows = {s: [0, 0, 0, []] for s in range(xy.shape[1])}   # frames, mais recuado, na área, distâncias
for half, first, last in half_video_blocks(GAME):
    k = KICKOFFS[half]
    print("tempo %d, kickoff no frame %d:" % (half, k), end="")
    # Que lado cada time defende: média de x no kickoff. < 0.5 -> defende o gol em x = 0.
    defends_left = {}
    for side, name in ((0, "casa"), (1, "fora")):
        sel = present_all[k] & (team == side)
        mean_x = np.nanmean(xy[k, sel, 0])
        defends_left[side] = mean_x < 0.5
        print("  %s x médio %.3f -> defende o gol %s" % (
            name, mean_x, "da esquerda (x=0)" if defends_left[side] else "da direita (x=1)"),
            end="")
    print()
    if defends_left[0] == defends_left[1]:
        print("  ATENÇÃO: os dois times parecem defender o mesmo lado; confira no vídeo.")

    block = slice(first, last + 1)
    x = xy[block, :, 0] * LENGTH
    y = xy[block, :, 1] * WIDTH
    present = present_all[block]
    for side in (0, 1):
        # distância até a própria linha de fundo
        dist = x if defends_left[side] else LENGTH - x
        mine = team == side
        d = np.where(present & mine, dist, np.inf)
        deepest = np.argmin(d, axis=1)                    # slot mais recuado em cada frame
        has_any = np.isfinite(d.min(axis=1))
        in_area = (dist <= AREA_DEPTH) & (np.abs(y - WIDTH / 2) <= AREA_WIDTH / 2)
        for s in np.flatnonzero(mine):
            p = present[:, s]
            rows[s][0] += int(p.sum())
            rows[s][1] += int(np.sum(p & has_any & (deepest == s)))
            rows[s][2] += int(np.sum(p & in_area[:, s]))
            rows[s][3].append(dist[p, s])

print("\n%-5s %-5s %8s %10s %9s %12s" % ("slot", "time", "frames", "+ recuado", "na área",
                                         "dist. med. m"))
table = []
for s, (n, deep, area, dists) in rows.items():
    if n < MIN_FRAMES:
        continue
    table.append((s, n, deep / n, area / n, float(np.median(np.concatenate(dists)))))
for s, n, deep, area, med in sorted(table, key=lambda r: -r[2]):
    print("%-5d %-5s %8d %9.1f%% %8.1f%% %12.1f" % (
        s + 1, "casa" if team[s] == 0 else "fora", n, 100 * deep, 100 * area, med))
print("\n(slots com menos de %d frames graduados omitidos)" % MIN_FRAMES)