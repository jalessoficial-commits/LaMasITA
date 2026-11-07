"""Rótulos para a análise de bola morta: GT público, rótulos fracos do jogo inteiro e testes.

    from rotulos import *          # com analise/ no sys.path

O GT público tem só 4 trechos de bola morta (5 min de jogo). Frames vizinhos são quase cópias um do
outro, então a amostra efetiva são esses 4 eventos, não 1469 frames. Daí as três peças daqui:

- weak_labels: rótulo fraco no jogo inteiro, tirado do Smart Tagging (SET PIECES) e da bola do ATD.
  Nunca lê GT; serve para treinar e calibrar, e o GT público fica só para testar.
- segment_ids: um id por trecho contíguo, para validação cruzada em blocos (GroupKFold).
- circular_shift_test: teste de permutação que desloca os rótulos em bloco, preservando a
  autocorrelação. Um teste por frame (Mann-Whitney em 7500 frames) dá p ~ 1e-300 para qualquer
  coisa e não significa nada.

Calibração (notebook 01, H0): nos 2 set pieces que caem no GT público, o start_frame fica DENTRO do
trecho morto, 6-12 s depois do apito e 3-12 s antes do reinício. Por isso a janela fraca de bola
morta é [start - PRE_S, start + POST_S], estreita e conservadora.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "viewer"), os.path.join(ROOT, "evaluation"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
ST_PATH = os.path.join(ROOT, "viewer", "data", "smart_tagging.json")
OUT_DIR = os.path.join(HERE, "out")

# Params
PRE_S, POST_S = 4.0, 2.0       # janela "morta provável" em torno do start_frame de um set piece
FAR_S = 30.0                   # "viva provável": a mais de 30 s de qualquer set piece ...
ALIVE_MIN_SPEED = 1.2          # ... e com o bloco em movimento (m/s, média de 3 s) ...
ALIVE_MIN_BALL_SPEED = 3.0     # ... ou com a bola andando (m/s, média de 1 s)
LINE_M = -1.0                  # bola a menos de 1 m da linha antes de sumir ...
LINE_GAP_S = 1.0               # ... por mais de 1 s: provável saída de bola


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def gt_public():
    """Rótulos H/A/D/X do GT público, indexados pelo frame do vídeo."""
    return pd.read_csv(GT_PATH)["label"].values.astype("<U1")


def runs(mask):
    """Lista de (início, fim_exclusivo) dos trechos True."""
    padded = np.concatenate(([False], np.asarray(mask, bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, ends))


def gt_dead_segments(gt=None):
    gt = gt_public() if gt is None else gt
    return runs(gt == "D")


def segment_ids(labels):
    """Um id inteiro por trecho contíguo de rótulo igual."""
    labels = np.asarray(labels)
    change = np.concatenate(([0], (labels[1:] != labels[:-1]).astype(int)))
    return np.cumsum(change)


def set_pieces():
    with open(ST_PATH, encoding="utf-8") as handle:
        events = json.load(handle)
    sp = [e for e in events if (e["code"] or "").upper() == "SET PIECES"]
    return pd.DataFrame(sp).sort_values("start_frame").reset_index(drop=True)


def weak_labels(X, index, pre_s=PRE_S, post_s=POST_S, far_s=FAR_S, use_line=True,
                alive_rule="jogadores"):
    """+1 = morta provável, 0 = viva provável, -1 = sem rótulo, no jogo inteiro e sem ler o GT.
    Retorna (labels, fonte) com fonte = 'set_piece' | 'linha' | 'longe' | ''.

    Circularidade: o rótulo herda as features que usa. alive_rule escolhe o que confirma a bola
    viva longe de reinícios -- "jogadores" (velocidade média 3 s > 1,2 m/s; 6% de D no GT),
    "bola" (bola a > 3 m/s; 8% de D) ou "nenhum" (13% de D). Treinar uma árvore com "jogadores" e
    deixar as features de velocidade dentro faz ela reaprender a regra; use_line idem para as
    features de lacuna da bola. O notebook 02 treina com alive_rule="bola", use_line=False e tira
    as features de velocidade da bola."""
    n = len(X)
    out = np.full(n, -1, dtype=np.int8)
    src = np.full(n, "", dtype="<U9")
    graded = graded_mask(n)

    to_next = np.nan_to_num(X[:, index["st_to_next_sp_s"]], nan=1e9)
    since = np.nan_to_num(X[:, index["st_since_sp_s"]], nan=1e9)
    far = (to_next > far_s) & (since > far_s)
    if alive_rule == "jogadores":
        far &= X[:, index["all_speed_mean_3s"]] > ALIVE_MIN_SPEED
    elif alive_rule == "bola":
        far &= X[:, index["ball_speed_1s"]] > ALIVE_MIN_BALL_SPEED
    out[far] = 0
    src[far] = "longe"

    if use_line:
        line = ((X[:, index["ball_last_out_dist_m"]] > LINE_M)
                & (X[:, index["ball_gap_since_s"]] > LINE_GAP_S))
        out[line] = 1
        src[line] = "linha"

    sp = set_pieces()
    for s in sp["start_frame"].values:
        lo, hi = max(0, int(s - pre_s * FPS)), min(n, int(s + post_s * FPS))
        out[lo:hi] = 1
        src[lo:hi] = "set_piece"

    out[~graded] = -1
    src[~graded] = ""
    return out, src


def restart_type(n_frames, pre_s=PRE_S, post_s=POST_S):
    """Tipo do set piece na janela fraca, '' fora dela."""
    out = np.full(n_frames, "", dtype="<U24")
    for s, t in set_pieces()[["start_frame", "type"]].values:
        out[max(0, int(s - pre_s * FPS)):int(s + post_s * FPS)] = t
    return out


# ---- estatística --------------------------------------------------------------------------------

def auc(y, x):
    """AUC de x separando y=1 de y=0 (Mann-Whitney), ignorando NaN. 0,5 = não separa."""
    from scipy.stats import rankdata
    ok = ~np.isnan(x)
    y, x = y[ok], x[ok]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def circular_shift_test(y, x, n=1000, min_shift_s=60, seed=0, stat=None):
    """p-valor de |stat(y, x) - 0,5| contra rótulos deslocados circularmente.

    Deslocar o vetor de rótulos inteiro preserva a estrutura temporal dos dois lados (duração dos
    trechos, autocorrelação da feature) e só destrói o alinhamento entre eles. É a hipótese nula
    certa para séries temporais: 'esta feature não acompanha a bola morta mais do que acompanharia
    trechos de mesma duração postos em lugar aleatório'.
    """
    stat = stat or auc
    rng = np.random.default_rng(seed)
    obs = stat(y, x)
    lo = int(min_shift_s * FPS)
    shifts = rng.integers(lo, max(lo + 1, len(y) - lo), n)
    null = np.array([stat(np.roll(y, s), x) for s in shifts])
    p = (1 + np.sum(np.abs(null - 0.5) >= abs(obs - 0.5))) / (n + 1)
    return obs, p, null


def cliffs_delta(a, b):
    """Cliff's δ = P(a > b) - P(a < b), robusto e sem supor normalidade. ±0,47 já é 'grande'."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if len(a) * len(b) > 4e6:
        rng = np.random.default_rng(0)
        a = rng.choice(a, min(len(a), 2000), replace=False)
        b = rng.choice(b, min(len(b), 2000), replace=False)
    d = a[:, None] - b[None, :]
    return float(np.mean(d > 0) - np.mean(d < 0))


def per_event_delta(x, events, alive_mask, context_s=30):
    """Cliff's δ de cada evento (início, fim) contra a bola viva nos context_s segundos em volta.

    Comparar cada trecho morto com a vizinhança dele elimina o efeito de região/minuto do jogo; a
    hipótese só se sustenta se o sinal do δ for o mesmo na maioria dos eventos.
    """
    out = []
    w = int(context_s * FPS)
    for s, e in events:
        lo, hi = max(0, s - w), min(len(x), e + w)
        ctx = np.zeros(len(x), bool)
        ctx[lo:hi] = True
        ctx[s:e] = False
        out.append(cliffs_delta(x[s:e], x[ctx & alive_mask]))
    return np.array(out)


def event_aligned(x, anchors, before_s=15, after_s=15):
    """Matriz (n_eventos, janela) de x alinhado em cada âncora; NaN fora do vídeo. Eixo em s."""
    b, a = int(before_s * FPS), int(after_s * FPS)
    t = np.arange(-b, a) / FPS
    M = np.full((len(anchors), b + a), np.nan, dtype=np.float32)
    for i, f in enumerate(anchors):
        lo, hi = f - b, f + a
        src_lo, src_hi = max(lo, 0), min(hi, len(x))
        M[i, src_lo - lo:src_hi - lo] = x[src_lo:src_hi]
    return t, M


def bootstrap_band(M, n=500, q=(2.5, 97.5), seed=0):
    """Média e banda de bootstrap POR EVENTO (reamostra linhas, não frames)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(M), (n, len(M)))
    means = np.array([np.nanmean(M[i], axis=0) for i in idx])
    return np.nanmean(M, axis=0), np.nanpercentile(means, q[0], axis=0), np.nanpercentile(means, q[1], axis=0)
