"""Eventos que abrem e fecham uma parada (M2 e M3 do detector de paradas).

    from eventos import Sinais, eventos_inicio, eventos_reinicio

Uma bola morta é um intervalo [início, reinício], e cada ponta tem o seu melhor sensor:

início      apito (audio_apito.py) -- faltas, impedimentos, gols; erro ~0,5 s
            saída da bola -- a bola some, ou fica presa na linha, depois de estar a < 2 m de uma
                             linha ou indo para fora. Laterais, escanteios e tiros de meta não são
                             apitados, e o .bin do ATD corta a bola na linha, então é o sumiço ou a
                             parada NA linha que denuncia a saída.
            colapso -- o detector v1 (velocidade dos próximos 5 s < limiar), o recurso final.

reinício    aceleração coletiva -- mínimo da velocidade média (1 s) seguido de um ganho forte nos
                             3 s seguintes. É a ponta precisa (−0,4 a +3 s no GT).
            Smart Tagging -- start_frame + deslocamento típico do tipo (calibrado aqui contra a
                             aceleração), quando o set piece está marcado.

Cada evento é um dict {frame, fonte, tipo, forca, ...}; nada aqui lê ground truth.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "viewer"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from features import LENGTH, WIDTH, forward_fill, load_atd  # noqa: E402
import rotulos as R  # noqa: E402
import arvore as T  # noqa: E402

FPS = 25
GOAL_HALF = 7.32 / 2
GOAL_AREA_HALF = 18.32 / 2

# Params
LINE_M = 2.0               # bola "na linha": a menos de 2 m de uma linha
STUCK_M = 0.3              # o .bin corta a bola na linha: |distância para fora| < 0,3 m = colada na linha
STUCK_S = 0.6              # ... por pelo menos 0,6 s
ALTURA_COS = 0.3           # bola só do detector: cruzamento cuja normal tem cos > 0,3 com o raio da câmera
EXIT_LOOKAHEAD_S = 0.4    # extrapola a última velocidade da bola por 0,4 s
GAP_MIN_S = 1.0            # o sumiço só conta se durar mais de 1 s
MIN_EVENT_GAP_S = 3.0      # dois eventos de início do mesmo tipo a < 3 s viram um
RESTART_GAIN = 1.0         # m/s: ganho de velocidade (1 s) nos 3 s depois do mínimo
RESTART_MIN_SPEED = 1.6    # m/s: o mínimo tem de ser lento de verdade
RESTART_HALF_WIN_S = 3.0   # o mínimo é mínimo local em ±3 s


class Sinais:
    """Tudo o que os eventos e a máquina de estados leem, calculado uma vez."""

    def __init__(self, X, I, F=None, I2=None, ball=None):
        if F is None:
            F, I2 = T.stack(X, I, T.context_features(X, I))
        self.F, self.I = F, I2
        self.N = len(F)
        self.graded = R.graded_mask(self.N)
        players, atd_ball, slot_teams, _ = load_atd()
        self.ball = atd_ball if ball is None else ball      # (n, 2) em metros; fontes_bola.py
        self.players = players
        self.speed1 = pd.Series(F[:, I2["all_speed_mean"]]).rolling(FPS, center=True, min_periods=1).mean().values
        try:
            from audio_apito import load_apito
            self.apito_snr, self.apito_meta = load_apito()
        except FileNotFoundError:
            self.apito_snr, self.apito_meta = None, {"eventos": []}

    def f(self, name):
        return self.F[:, self.I[name]]


# ---- classificação pela posição da bola -----------------------------------------------------------

def zona_da_bola(x, y):
    """Que reinício a posição (x, y) sugere. Usada na última posição conhecida antes do sumiço."""
    if np.isnan(x):
        return "desconhecida"
    d_goal = min(x, LENGTH - x)
    d_touch = min(y, WIDTH - y)
    if d_goal < LINE_M and d_goal <= d_touch:
        return "linha_de_fundo"            # escanteio ou tiro de meta: decide-se depois
    if d_touch < LINE_M:
        return "lateral"
    if np.hypot(x - LENGTH / 2, y - WIDTH / 2) < 3:
        return "centro"
    return "campo"


# ---- início ----------------------------------------------------------------------------------------

def eventos_apito(S, min_snr=10.0):
    out = []
    for fi, fe, snr in S.apito_meta.get("eventos", []):
        if 0 <= fi < S.N and S.graded[fi] and snr >= min_snr:
            out.append(dict(frame=int(fi), fonte="apito", tipo="apito", forca=float(snr), fim_evento=int(fe)))
    return out


def eventos_saida_bola(S, fora=False):
    """A bola some depois de estar na linha, ou fica colada na linha (o .bin do ATD a corta ali).

    fora=True troca "colada" por "na linha ou além dela": é a regra certa para uma bola sem corte,
    que passa da linha em vez de ficar nela (fontes_bola.py)."""
    x, y = S.ball[:, 0], S.ball[:, 1]
    present = ~np.isnan(x)
    out_dist = np.maximum.reduce([-x, x - LENGTH, -y, y - WIDTH])     # >0 fora; ~0 colada
    vel = np.full_like(S.ball, np.nan)
    vel[1:] = (S.ball[1:] - S.ball[:-1]) * FPS
    vel = pd.DataFrame(vel).rolling(5, min_periods=2).mean().values
    ev = []

    # (a) sumiço: início de cada lacuna de bola > GAP_MIN_S
    for s, e in R.runs(~present):
        if s == 0 or e - s < GAP_MIN_S * FPS or not S.graded[s]:
            continue
        lx, ly = x[s - 1], y[s - 1]
        px, py = lx + vel[s - 1, 0] * EXIT_LOOKAHEAD_S, ly + vel[s - 1, 1] * EXIT_LOOKAHEAD_S
        d_line = -out_dist[s - 1]
        going_out = (px < 0) or (px > LENGTH) or (py < 0) or (py > WIDTH)
        if d_line < LINE_M or going_out:
            zx, zy = (px, py) if going_out and not np.isnan(px) else (lx, ly)
            zx, zy = np.clip(zx, 0, LENGTH), np.clip(zy, 0, WIDTH)
            ev.append(dict(frame=int(s), fonte="saida_bola", tipo=zona_da_bola(zx, zy),
                           forca=float(LINE_M - min(d_line, LINE_M)) + 2.0 * going_out,
                           x=float(lx), y=float(ly), lacuna_s=(e - s) / FPS))

    # (b) colada na linha: o corte do .bin deixa a bola em out_dist ~ 0 enquanto ela está fora
    stuck = present & (out_dist > -STUCK_M if fora else np.abs(out_dist) < STUCK_M)
    if fora and getattr(S, "ball_raio", None) is not None:
        # bola só do detector (fontes_bola 'fundida'): uma bola no ar aparece deslocada ao longo do raio
        # da câmera, para longe dela. Um cruzamento na direção desse raio pode ser só altura: não conta.
        normal = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], float)[
            np.nanargmax(np.nan_to_num(np.c_[-x, x - LENGTH, -y, y - WIDTH], nan=-1e9), axis=1)]
        ao_longo = np.nan_to_num(np.sum(normal * S.ball_raio, axis=1))
        stuck &= ~((S.ball_qualidade == "detector") & (ao_longo > ALTURA_COS))
    for s, e in R.runs(stuck):
        if e - s >= STUCK_S * FPS and S.graded[s]:
            ev.append(dict(frame=int(s), fonte="saida_bola", tipo=zona_da_bola(x[s], y[s]),
                           forca=2.0, x=float(x[s]), y=float(y[s]), lacuna_s=0.0))
    return dedup(sorted(ev, key=lambda d: d["frame"]))


def dedup(ev, gap_s=MIN_EVENT_GAP_S):
    out = []
    for e in ev:
        if out and e["frame"] - out[-1]["frame"] < gap_s * FPS and e["fonte"] == out[-1]["fonte"]:
            if e["forca"] > out[-1]["forca"]:
                out[-1] = dict(e, frame=out[-1]["frame"])
            continue
        out.append(e)
    return out


def eventos_inicio(S, usar=("apito", "saida_bola"), fora=False):
    ev = []
    if "apito" in usar:
        ev += eventos_apito(S)
    if "saida_bola" in usar:
        ev += eventos_saida_bola(S, fora)
    return sorted(ev, key=lambda d: d["frame"])


# ---- reinício --------------------------------------------------------------------------------------

def eventos_aceleracao(S):
    """Mínimo local da velocidade (1 s) seguido de ganho forte nos 3 s seguintes."""
    s1 = S.speed1
    w = int(RESTART_HALF_WIN_S * FPS)
    local_min = pd.Series(s1).rolling(2 * w + 1, center=True, min_periods=1).min().values
    fut_max = pd.Series(s1[::-1]).rolling(3 * FPS, min_periods=1).max().values[::-1]
    cand = np.flatnonzero((s1 <= local_min + 1e-6) & (s1 < RESTART_MIN_SPEED)
                          & (fut_max - s1 > RESTART_GAIN) & S.graded)
    out = []
    for c in cand:
        if out and c - out[-1]["frame"] < w:
            continue
        out.append(dict(frame=int(c), fonte="aceleracao", tipo="reinicio",
                        forca=float(fut_max[c] - s1[c]), vel_min=float(s1[c])))
    return out


def calibrar_offset_smart_tagging(S, acel=None, janela_s=(-5, 20)):
    """Para cada tipo de set piece: mediana (s) entre o start_frame e a aceleração seguinte."""
    acel = acel if acel is not None else eventos_aceleracao(S)
    af = np.array([a["frame"] for a in acel])
    rows = []
    for _, e in R.set_pieces().iterrows():
        lo, hi = e.start_frame + janela_s[0] * FPS, e.start_frame + janela_s[1] * FPS
        k = af[(af >= lo) & (af <= hi)]
        rows.append(dict(tipo=e.type, start_frame=e.start_frame,
                         offset_s=(k[0] - e.start_frame) / FPS if len(k) else np.nan))
    d = pd.DataFrame(rows)
    med = d.groupby("tipo").offset_s.median()
    return med.fillna(d.offset_s.median()).to_dict(), d


def eventos_smart_tagging(S, offsets):
    out = []
    for _, e in R.set_pieces().iterrows():
        f = int(e.start_frame + offsets.get(e.type, 3.0) * FPS)
        if 0 <= f < S.N and S.graded[f]:
            out.append(dict(frame=f, fonte="smart_tagging", tipo=e.type, forca=10.0, start_frame=int(e.start_frame)))
    return out


def eventos_reinicio(S, usar=("aceleracao", "smart_tagging"), offsets=None):
    ev = []
    acel = eventos_aceleracao(S)
    if "aceleracao" in usar:
        ev += acel
    if "smart_tagging" in usar:
        offsets = offsets or calibrar_offset_smart_tagging(S, acel)[0]
        ev += eventos_smart_tagging(S, offsets)
    return sorted(ev, key=lambda d: d["frame"])
