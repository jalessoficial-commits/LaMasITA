"""Exporta para o viewer o resultado de comparar_tracking.py (rode-o antes).

    python analise/exportar_viewer_tracking.py

Grava em viewer/data/:

    bola_<fonte>.csv           camada de bola (frame,track_id,team,x,y) -> "➕ Adicionar tracking (CSV)"
    pred_<variante>.csv        baseline_smooth125 + D da variante (frame,label) -> "🎯 Carregar predição (CSV)"
    eventos_<variante>.json    paradas e apitos da variante -> http://localhost:8000/viewer/?eventos=eventos_<variante>
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import fontes_bola as B  # noqa: E402
import paradas as P  # noqa: E402
import rotulos as R  # noqa: E402
from audio_apito import load_apito  # noqa: E402
from features import LENGTH, WIDTH  # noqa: E402

SRC = os.path.join(HERE, "out", "tracking")
DST = os.path.join(os.path.dirname(HERE), "viewer", "data")


class _Apitos:
    apito_meta = load_apito()[1]


def main():
    for fonte in B.FONTES[1:]:                       # atd_bin já é a camada ATD do viewer
        ball = B.carregar_bola(fonte)
        f = np.flatnonzero(~np.isnan(ball[:, 0]))
        path = os.path.join(DST, "bola_{}.csv".format(fonte))
        pd.DataFrame({"frame": f, "track_id": fonte, "team": "B",
                      "x": np.round(ball[f, 0] / LENGTH, 6), "y": np.round(ball[f, 1] / WIDTH, 6)}).to_csv(path, index=False)
        print(path)

    base = T.load_baseline("baseline_smooth125")
    for d_path in sorted(glob.glob(os.path.join(SRC, "D_*.csv"))):
        nome = os.path.basename(d_path)[2:-4]
        d = pd.read_csv(d_path).morta.values.astype(bool)
        labels = T.combine_with_baseline(base, d, R.graded_mask(len(d)))
        pd.DataFrame({"frame": np.arange(len(d)), "label": labels}).to_csv(
            os.path.join(DST, "pred_{}.csv".format(nome)), index=False)
        tab = pd.read_csv(os.path.join(SRC, "paradas_{}.csv".format(nome)))
        P.exportar_viewer(tab, _Apitos, os.path.join(DST, "eventos_{}.json".format(nome)))
        print(nome)


if __name__ == "__main__":
    main()
