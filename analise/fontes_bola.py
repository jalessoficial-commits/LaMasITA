"""Fontes da posição da bola para a análise de bola morta.

    from fontes_bola import FONTES, carregar_bola
    ball = carregar_bola("kalman_pred")        # (n_frames, 2) em metros, NaN = sem bola

Índice 0 = frame 0 do vídeo, como em viewer/features.py.

atd_bin         viewer/data/atd_tactical.bin. O preprocess.py corta as coordenadas em [0, 65534/65000],
                então uma bola fora do campo vira uma bola em x = 0 ou y = 0, ou a +0,9 m / +0,6 m do
                outro lado. A "bola presa na linha" do ATD é esse corte.
atd_bruto       a bola do ATD lida do .txt, sem o corte.
tracker         fusão ATD + detector (tracking_bola), sem Kalman.
kalman_filter   Kalman do tracker, só nos frames em que a medição foi aceita.
kalman_pred     Kalman do tracker, previsão de 1 frame. É a fonte com mais cobertura porque continua
                extrapolando por até 1 s (25 frames) depois da última medição aceita.
fundida         "ATD primeiro": a bola do ATD bruto onde ela existe; o detector do tracker só onde o
                ATD não tem bola; buracos curtos do ATD (<= INTERP_MAX_S) interpolados entre as pontas.
                extras_fundida() devolve a qualidade de cada frame e o raio da câmera no solo.

Por que o ATD primeiro: o tracker projeta a bola da imagem no gramado pela homografia, então uma bola
no ar aparece mais longe da câmera, ao longo do raio dela. Nos frames em que os dois têm bola, 90-95%
dos desvios > 3 m caem ao longo desse raio, para longe da câmera (detector_trocou: 6 m de mediana).
Nos frames só do detector não há como separar bola no ar de bola no chão (âncoras do ATD, pontas da
ilha do detector e curvatura ao longo do raio foram testados contra o ATD escondido e não ajudaram),
então a qualidade 'detector' fica marcada e as regras decidem o que fazer com ela.

Os CSVs do Kalman vêm de ball_kalman/kalman_ball.py, no repositório do desafio (HACKATON_DIR).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "viewer")):
    if p not in sys.path:
        sys.path.insert(0, p)
from features import LENGTH, WIDTH, load_atd  # noqa: E402
from games import asset_paths  # noqa: E402

HACKATON_DIR = os.environ.get("HACKATON_DIR", r"C:\Users\NunoP\Desktop\Hackaton_barca")
CSV = {
    "tracker": os.path.join("viewer", "data", "ball_tracker.csv"),
    "kalman_filter": os.path.join("ball_kalman", "output", "tracker_kalman_filter.csv"),
    "kalman_pred": os.path.join("ball_kalman", "output", "tracker_kalman_pred.csv"),
}
FONTES = ("atd_bin", "atd_bruto") + tuple(CSV) + ("fundida",)
# CSV do tracking_bola/export_ball.py COM a coluna `fonte` (o do viewer do Hackaton não tem)
TRACKER_FONTES = [os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_ball_tracker.csv"),
                  os.path.join(os.path.expanduser("~"), "Downloads", "BAR-ATH_tactical_ball_tracker.csv")]
HOMOGRAPHY_BIN = os.path.join(ROOT, "viewer", "data", "homography_tactical.bin")
INTERP_MAX_S = 1.5         # buraco do ATD até 1,5 s: interpolação linear (melhor que o detector nos testes)
FPS = 25
_cache = {}


def _atd_bruto(n_frames):
    out = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(asset_paths()["atd_raw"]) as handle:
        for line in handle:
            frame_part, _, rest = line.partition(":")
            bola = rest.rpartition(":")[2].split(",")
            f = int(frame_part) - 1
            if 0 <= f < n_frames and bola[0] != "NaN" and bola[1] != "NaN":
                out[f] = float(bola[0]) * LENGTH, float(bola[1]) * WIDTH
    return out


def raio_camera(xy, frames, delta_px=5.0):
    """Direção unitária, no solo, para onde uma bola no ar aparece deslocada (para longe da câmera).

    xy: (n, 2) em metros nos frames `frames`. A homografia leva o campo normalizado (0-1) a pixels; a
    direção é a do ponto que fica delta_px acima na imagem."""
    H = np.fromfile(HOMOGRAPHY_BIN, dtype=np.float32).reshape(-1, 3, 3).astype(float)[frames]
    q = np.c_[xy[:, 0] / LENGTH, xy[:, 1] / WIDTH, np.ones(len(xy))]
    p = np.einsum("nij,nj->ni", H, q)
    px = p[:, :2] / p[:, 2:] - [0.0, delta_px]
    r = np.einsum("nij,nj->ni", np.linalg.inv(H), np.c_[px, np.ones(len(px))])
    up = r[:, :2] / r[:, 2:] * [LENGTH, WIDTH]
    u = up - xy
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def _fundida(n_frames):
    path = next((p for p in TRACKER_FONTES if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("CSV do tracker com a coluna fonte: " + " ou ".join(TRACKER_FONTES))
    atd = _atd_bruto(n_frames).astype(float)
    ball = atd.copy()
    qual = np.where(np.isnan(atd[:, 0]), "", "atd").astype("<U8")

    has = ~np.isnan(atd[:, 0])                         # buracos curtos: interpolação entre as pontas
    pad = np.concatenate(([False], ~has, [False]))
    starts, ends = np.flatnonzero(~pad[:-1] & pad[1:]), np.flatnonzero(pad[:-1] & ~pad[1:])
    for s, e in zip(starts, ends):
        if 0 < s and e < n_frames and e - s <= INTERP_MAX_S * FPS:
            t = ((np.arange(s, e) - (s - 1)) / (e - (s - 1)))[:, None]
            ball[s:e] = atd[s - 1] + (atd[e] - atd[s - 1]) * t
            qual[s:e] = "interp"

    t = pd.read_csv(path)
    t = t[(t.fonte == "detector") & t.x.notna() & (t.frame >= 0) & (t.frame < n_frames)]
    f = t.frame.values
    f = f[qual[f] == ""]                               # o detector só onde o ATD não tem bola
    xy = t.set_index("frame").loc[f, ["x", "y"]].values * np.array([LENGTH, WIDTH])
    ball[f] = xy
    qual[f] = "detector"
    u = np.full((n_frames, 2), np.nan)
    ok = np.flatnonzero(qual != "")
    u[ok] = raio_camera(ball[ok], ok)
    return ball.astype(np.float32), qual, u


def extras_fundida():
    """(qualidade, raio): qualidade por frame em {'atd', 'interp', 'detector', ''} e o raio da câmera
    no solo (n, 2) -- veja raio_camera."""
    if "fundida" not in _cache:
        _cache["fundida"] = _fundida(load_atd()[3]["n_frames"])
    return _cache["fundida"][1], _cache["fundida"][2]


def carregar_bola(nome):
    _, atd, _, meta = load_atd()
    n_frames = meta["n_frames"]
    if nome == "atd_bin":
        return atd
    if nome == "atd_bruto":
        return _atd_bruto(n_frames)
    if nome == "fundida":
        extras_fundida()
        return _cache["fundida"][0]
    table = pd.read_csv(os.path.join(HACKATON_DIR, CSV[nome]))
    table = table[(table.frame >= 0) & (table.frame < n_frames)]
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[table.frame.values] = table[["x", "y"]].values * np.array([LENGTH, WIDTH])
    return ball
