"""Detector de paradas (analise/paradas.py + analise/reinicio.py) em outros jogos, com cada preditor de posse.

    DFL_DIR=... METRICA_DIR=... python validacao_metrica/testar_detector_paradas.py J03WN1 J03WOH J03WOY metrica1 metrica2

Para cada jogo e semente, três D (o detector atual, o detector + reinicio.aplicar, e o D do gabarito) com
cinco preditores de posse (baseline, posse/predictor.py com e sem portão, analise/posse.py com custo 25 e 50).
Também grava as métricas do próprio detector: paradas encontradas (IoU >= 0,5), paradas previstas corretas,
erro e viés no início e no fim.

Gabarito: Bundesliga (dfl.py), BallStatus por frame; Metrica (metrica.py), paradas montadas pelos eventos.

O detector roda como na configuração da equipe (bola bruta, regra "fora", v1_uniao). O que estes jogos não
têm é emulado:
    bola              lacunas aleatórias até faltar em 41% dos frames avaliados (como o ATD do BAR-ATH);
                      x reescalado para o campo de 105,3 m do features.py
    movimento         velocidade média dos jogadores e jogadores nas áreas, com as fórmulas do viewer/features.py
    apito             nas paradas que não começam com a bola fora: no início ± 0,5 s, 90% detectados, mais 10
                      apitos falsos por jogo em jogo vivo
    Smart Tagging     em 55% das paradas (59 tags em ~110 paradas no BAR-ATH): start_frame = reinício - U(3, 12) s,
                      tipo pela posição da bola
    v1                só o nó N1 (velocidade dos próximos 5 s < 1,1 m/s), o caminho do código sem arvore_D.csv;
                      a árvore completa é ajustada no Smart Tagging do BAR-ATH
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "viewer"), os.path.join(ROOT, "analise"),
          os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "posse")):
    if p not in sys.path:
        sys.path.insert(0, p)

import arvore as T  # noqa: E402
import dfl  # noqa: E402
import eventos as E  # noqa: E402
import metrica as M  # noqa: E402
import paradas as P  # noqa: E402
import reinicio as RE  # noqa: E402
import rotulos as R  # noqa: E402
from features import (LENGTH, MAX_PLAYER_SPEED, PENALTY_DEPTH, PENALTY_WIDTH, VEL_HALF_WINDOW, WIDTH,  # noqa: E402
                      centred_velocity)
from testar_dono_reinicio import combine_with_baseline, degradar, mais_proximo, pontuar, smooth  # noqa: E402
from testar_predictor_samuel import carregar_nuno, samuel  # noqa: E402

FPS = 25
SEMENTES = (0, 1)
OUT = os.path.join(HERE, "out")


class SinaisEmulados:
    """O que paradas.detectar_paradas, eventos.* e reinicio.aplicar leem de eventos.Sinais."""

    def __init__(self, J, ball, apitos):
        self.N = J.N
        self.graded = J.gt != "X"
        self.players = J.players
        self.ball = ball
        vel = centred_velocity(J.players, VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
        with np.errstate(all="ignore"):
            speed = np.nanmean(np.hypot(vel[..., 0], vel[..., 1]), axis=1)
        x, y = J.players[..., 0], J.players[..., 1]
        in_box = (np.minimum(x, LENGTH - x) < PENALTY_DEPTH) & (np.abs(y - WIDTH / 2) < PENALTY_WIDTH / 2)
        self._f = {"all_speed_mean": speed,
                   "speed_fut5s_mean": T._roll(speed, 5 * FPS, "mean", "f"),
                   "all_players_in_boxes": np.sum(in_box, axis=1).astype(float)}
        self.speed1 = pd.Series(speed).rolling(FPS, center=True, min_periods=1).mean().values
        self.apito_snr = None
        self.apito_meta = {"eventos": [[int(f), int(f) + 5, 20.0] for f in apitos]}

    def f(self, name):
        return self._f[name]


def carregar(nome):
    """'J03WN1' etc. (DFL) ou 'metrica1' / 'metrica2'."""
    J = M.carregar_jogo(int(nome[-1])) if nome.startswith("metrica") else dfl.carregar_jogo_dfl(nome)
    J.players = J.players.copy()
    J.ball = J.ball.copy()
    J.players[..., 0] *= LENGTH / 105.0
    J.ball[:, 0] *= LENGTH / 105.0
    return J


def tipo_parada(J, ini, fim):
    """Pela bola: ('lateral' | 'linha_de_fundo' | 'apito', tipo de set piece para a tag emulada)."""
    a, b = max(0, ini - FPS), min(J.N, ini + 2 * FPS)
    x, y = J.ball[a:b, 0], J.ball[a:b, 1]
    with np.errstate(all="ignore"):
        out_x = np.nanmax(np.maximum(-x, x - LENGTH)) if np.isfinite(x).any() else -1
        out_y = np.nanmax(np.maximum(-y, y - WIDTH)) if np.isfinite(y).any() else -1
    rx, ry = J.ball[min(fim, J.N - 1)]
    if out_y > 0 and out_y >= out_x:
        return "lateral", "Throw-in"
    if out_x > 0:
        canto = min(rx, LENGTH - rx) < 3 and min(ry, WIDTH - ry) < 3
        return "linha_de_fundo", "Corner Kick" if canto else "Goal Kick"
    centro = np.hypot(rx - LENGTH / 2, ry - WIDTH / 2) < 3
    return "apito", "Kick Off (After Goal)" if centro else "Free Kick"


def emular(J, seed, recall_apito=0.9, falsos=10, frac_tags=0.55):
    rng = np.random.default_rng(seed)
    apitos, tags = [], []
    for r in J.reinicios.itertuples():
        quebra, tipo_tag = tipo_parada(J, r.inicio, r.fim)
        if quebra == "apito" and rng.random() < recall_apito:
            apitos.append(r.inicio + int(rng.normal(0, 0.5) * FPS))
        if rng.random() < frac_tags:
            tags.append(dict(start_frame=max(r.inicio, r.fim - int(rng.uniform(3, 12) * FPS)),
                             type=tipo_tag, code="SET PIECES"))
    apitos += list(rng.choice(np.flatnonzero(np.isin(J.gt, ["H", "A"])), falsos, replace=False))
    return sorted(apitos), pd.DataFrame(tags).sort_values("start_frame").reset_index(drop=True)


def detectar(J, ball, apitos, tags):
    """(D atual, D com reinicio.aplicar), na configuração da equipe."""
    S = SinaisEmulados(J, ball, apitos)
    cfg = P.Config(v1_uniao=True)
    R.set_pieces = lambda: tags                      # eventos.* leem os set pieces daqui
    v1 = T.clean_runs((S.f("speed_fut5s_mean") < cfg.v1_limiar) & S.graded,
                      cfg.v1_min_dead_s, cfg.v1_min_alive_s) & S.graded
    inicio = E.eventos_inicio(S, cfg.usar_inicio, fora=True)
    offsets = E.calibrar_offset_smart_tagging(S)[0] if len(tags) else {}
    dead, tab = P.detectar_paradas(S, cfg, inicio=inicio, offsets=offsets, v1_dead=v1)
    novo, _ = RE.aplicar(S, tab, inicio, cfg)
    return dead & S.graded, novo


def metricas(gt, dead):
    """Frame a frame e por parada (pareamento por IoU >= 0,5)."""
    g = gt != "X"
    tD, pD = (gt == "D") & g, dead & g
    tp = (tD & pD).sum()
    out = dict(prec_frame=tp / max(1, pD.sum()), rec_frame=tp / max(1, tD.sum()))
    tr, pr = T.runs(tD), T.runs(pD)
    usados, err_i, err_f = set(), [], []
    for s, e in tr:
        best, bi = 0, None
        for k, (ps, pe) in enumerate(pr):
            if pe > s and ps < e and k not in usados:
                iou = (min(e, pe) - max(s, ps)) / (max(e, pe) - min(s, ps))
                if iou > best:
                    best, bi = iou, k
        if best >= 0.5:
            usados.add(bi)
            err_i.append((pr[bi][0] - s) / FPS)
            err_f.append((pr[bi][1] - e) / FPS)
    out.update(n_gt=len(tr), n_pred=len(pr), paradas_encontradas=len(err_i) / max(1, len(tr)),
               previstas_corretas=len(err_i) / max(1, len(pr)),
               erro_inicio_s=np.median(np.abs(err_i)) if err_i else np.nan,
               erro_fim_s=np.median(np.abs(err_f)) if err_f else np.nan,
               vies_fim_s=np.median(err_f) if err_f else np.nan)
    return out


def rodar(nome, PO):
    J = carregar(nome)
    g = J.gt != "X"
    linhas, det = [], []
    for s in SEMENTES:
        ball = degradar(J.ball, g, alvo=0.41, seed=s)
        apitos, tags = emular(J, s)
        atual, novo = detectar(J, ball, apitos, tags)
        raw = mais_proximo(J, ball)
        base = smooth(raw)
        dists = PO.distancias(ball, J.players, J.time_slot) if PO else None
        for dn, dead in (("D atual", atual), ("D com reinicio.py", novo), ("D do gabarito", (J.gt == "D") & g)):
            if dn != "D do gabarito":
                det.append(dict(jogo=nome, semente=s, D=dn, **metricas(J.gt, dead)))
            r = {"baseline": pontuar(J.gt, combine_with_baseline(base, dead, g))[0],
                 "Samuel": pontuar(J.gt, samuel(J, ball, dead, gate=True))[0],
                 "Samuel sem portão": pontuar(J.gt, samuel(J, ball, dead, gate=False))[0]}
            if PO is not None:
                for c in (25.0, 50.0):
                    r["Nuno custo {:.0f}".format(c)] = pontuar(J.gt, PO.prever(
                        ball, J.players, J.time_slot, dead, g, PO.Config(custo_troca=c), dists=dists))[0]
            for k, v in r.items():
                linhas.append(dict(jogo=nome, semente=s, D=dn, preditor=k, spellF1=v))
            print(nome, s, dn, {k: round(v, 3) for k, v in r.items()}, flush=True)
    return pd.DataFrame(linhas), pd.DataFrame(det)


def main():
    jogos = sys.argv[1:] or ["J03WN1", "J03WOH", "J03WOY", "metrica1", "metrica2"]
    os.makedirs(OUT, exist_ok=True)
    PO = carregar_nuno()
    res, det = zip(*(rodar(j, PO) for j in jogos))
    res, det = pd.concat(res), pd.concat(det)
    res.to_csv(os.path.join(OUT, "detector_paradas_spell.csv"), index=False)
    det.to_csv(os.path.join(OUT, "detector_paradas_metricas.csv"), index=False)
    print()
    print(res.pivot_table(index="preditor", columns="D", values="spellF1").round(3).to_string())
    print()
    print(det.groupby("D").mean(numeric_only=True).drop(columns="semente").round(3).to_string())


if __name__ == "__main__":
    main()
