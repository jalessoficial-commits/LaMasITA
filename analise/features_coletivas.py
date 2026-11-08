"""Features de comportamento coletivo para a decisão bola viva / morta.

    python viewer/features.py                 # uma vez (gera features_tactical.*)
    python analise/features_coletivas.py      # ~1-3 min; grava viewer/data/features_coletivas.*

    from features_coletivas import load_coletivas
    C, cmeta = load_coletivas()               # C[frame, cmeta["index"]["chase_cos_k3_all"]]

Mesma convenção de viewer/features.py: índice 0 = frame 0 do vídeo, metros e segundos, NaN = sem
valor, nada aqui lê ground truth. Cada grupo traduz uma hipótese de outra área para o futebol:

quimiotaxia (perseguição)    jogadores perto da bola se movem em direção a ela enquanto a bola está
                             viva, como bactérias subindo um gradiente; com a bola morta, não.
ordem coletiva (Vicsek)      polarização, ordem nemática, correlação de velocidade entre vizinhos
                             (Cavagna et al. 2010) e "temperatura" (energia cinética).
mobilidade (MSD)             deslocamento quadrático médio e expoente α: ≈2 balístico (corrida
                             dirigida), ≈1 difusivo, <1 confinado (parados esperando o reinício).
forma em mudança             contração/expansão radial, taxa de variação do comprimento e do stretch.
contexto passado             mínimo / máximo / tendência dos sinais-chave em janelas que só olham o
                             passado, para a árvore poder decidir online, frame a frame.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "viewer")):
    if p not in sys.path:
        sys.path.insert(0, p)
from features import (FPS, LENGTH, WIDTH, VEL_HALF_WINDOW, BALL_HALF_WINDOW,  # noqa: E402
                      MAX_PLAYER_SPEED, MAX_BALL_SPEED, OUT_DIR, centred_velocity, describe,
                      forward_fill, load_atd, load_features, rolling_mean, rolling_std,
                      run_lengths)
from games import half_video_blocks, GAME  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Params
MOVING = 0.5               # m/s; abaixo disso o jogador não tem direção de corrida
BALL_FFILL_S = 2.0         # mantém a última posição da bola por até 2 s de lacuna
NEIGHBOR_R = 10.0          # m; raio da correlação de velocidade entre vizinhos
MSD_LAGS_S = (0.2, 1.0, 2.0)
STILL_SPEED = 1.0          # m/s; bloco "parado" (média de 1 s)
CHUNK = 4000               # frames por bloco nas contas O(jogadores²)
PAST_WINDOWS_S = (2, 5)
LINE_NEAR_M = -2.0         # bola a menos de 2 m da linha


def limited_ffill(a, max_frames):
    """forward_fill que só cobre lacunas de até max_frames frames (a: (n,) ou (n, d))."""
    a2 = a if a.ndim == 2 else a[:, None]
    valid = ~np.isnan(a2[:, 0])
    idx = np.where(valid, np.arange(len(a2)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a2[idx].copy()
    age = np.arange(len(a2)) - idx
    out[(age > max_frames) | (~valid[idx])] = np.nan
    return out if a.ndim == 2 else out[:, 0]


def past_rolling(a, window, how):
    s = pd.Series(a)
    r = s.rolling(window, min_periods=max(1, window // 4))
    return getattr(r, how)().values.astype(np.float32)


class Bag:
    def __init__(self, n):
        self.n, self.items = n, []

    def add(self, name, values, unit, group, desc):
        v = np.asarray(values, dtype=np.float32)
        assert v.shape == (self.n,), (name, v.shape)
        self.items.append((name, v, unit, group, desc))


# ---- grupos ---------------------------------------------------------------------------------------

def chase_features(bag, players, vel, slot_teams, ball):
    """Perseguição: projeção da velocidade de cada jogador na direção da bola."""
    print("  perseguição ...")
    b = limited_ffill(ball, int(BALL_FFILL_S * FPS))
    dx = b[:, None, 0] - players[..., 0]
    dy = b[:, None, 1] - players[..., 1]
    dist = np.hypot(dx, dy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = dx / dist, dy / dist
    speed = np.hypot(vel[..., 0], vel[..., 1])
    toward = vel[..., 0] * ux + vel[..., 1] * uy          # m/s, >0 = indo para a bola
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(speed > MOVING, toward / speed, 0.0)  # parado não persegue: 0
    cos[np.isnan(toward)] = np.nan
    has_ball = ~np.isnan(b[:, 0])

    for prefix, mask in (("all", np.ones(len(slot_teams), bool)), ("h", slot_teams == "H"),
                         ("a", slot_teams == "A")):
        d = np.where(np.isnan(cos[:, mask]), np.inf, dist[:, mask])
        order = np.argsort(d, axis=1)
        for k in ((3, 5) if prefix == "all" else (3,)):
            sel = order[:, :k]
            c = np.take_along_axis(cos[:, mask], sel, axis=1)
            t = np.take_along_axis(toward[:, mask], sel, axis=1)
            ok = np.isfinite(np.take_along_axis(d, sel, axis=1))
            c[~ok], t[~ok] = np.nan, np.nan
            bag.add("chase_cos_k{}_{}".format(k, prefix), np.where(has_ball, np.nanmean(c, 1), np.nan),
                    "-1..1", "perseguicao",
                    "cosseno médio velocidade x direção da bola, {} mais próximos ({})".format(k, prefix))
            if prefix == "all":
                bag.add("approach_rate_k{}".format(k), np.where(has_ball, np.nanmean(t, 1), np.nan),
                        "m/s", "perseguicao",
                        "velocidade média de aproximação da bola dos {} mais próximos".format(k))
    valid = ~np.isnan(toward)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.sum(toward > MOVING, axis=1) / valid.sum(axis=1)
    bag.add("frac_closing", np.where(has_ball, frac, np.nan), "0-1", "perseguicao",
            "fração dos jogadores se aproximando da bola a > {} m/s".format(MOVING))
    bag.add("ball_nearest3_d_m", np.where(has_ball, np.nanmean(np.sort(dist, 1)[:, :3], 1), np.nan),
            "m", "perseguicao", "distância média dos 3 jogadores mais próximos à bola (bola mantida 2 s)")

    # o bloco "segue" a bola?
    cx = np.nanmean(players[..., 0], axis=1)
    cy = np.nanmean(players[..., 1], axis=1)
    cvel = centred_velocity(np.stack([cx, cy], 1), VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    bvel = centred_velocity(ball, BALL_HALF_WINDOW, MAX_BALL_SPEED)
    with np.errstate(invalid="ignore", divide="ignore"):
        bc = np.sum(cvel * bvel, 1) / (np.hypot(*cvel.T) * np.hypot(*bvel.T))
    bag.add("ball_centroid_cos_2s", rolling_mean(bc, 2 * FPS), "-1..1", "perseguicao",
            "cosseno médio (2 s) entre a velocidade da bola e a do centróide de todos")

    # bola perdida perto da linha
    out_dist = np.maximum.reduce([-ball[:, 0], ball[:, 0] - LENGTH, -ball[:, 1], ball[:, 1] - WIDTH])
    last_out = forward_fill(out_dist)
    gap_since, _ = run_lengths(np.isnan(ball[:, 0]))
    lost = np.where(last_out > LINE_NEAR_M, gap_since / FPS, 0.0)
    bag.add("ball_lost_at_line_s", lost, "s", "perseguicao",
            "há quanto tempo a bola sumiu, se sumiu a menos de 2 m da linha (0 caso contrário)")


def order_features(bag, players, vel, X, index):
    print("  ordem coletiva ...")
    speed = np.hypot(vel[..., 0], vel[..., 1])
    moving = speed > MOVING
    theta = np.arctan2(vel[..., 1], vel[..., 0])
    z2 = np.where(moving, np.exp(2j * theta), np.nan)
    bag.add("nematic_order", np.abs(np.nanmean(z2, axis=1)), "0-1", "ordem",
            "ordem nemática: alinhamento de eixo sem sentido (vai-e-volta conta)")

    pol = X[:, index["all_polarization"]]
    bag.add("polarization_susc_2s", rolling_std(pol, 2 * FPS) ** 2, "-", "ordem",
            "susceptibilidade: variância da polarização em 2 s (pico em transição de fase)")
    bag.add("polarization_1s", rolling_mean(pol, FPS), "0-1", "ordem", "polarização, média de 1 s")

    ke = 0.5 * np.nanmean(speed ** 2, axis=1)
    bag.add("kinetic_energy", ke, "m2/s2", "ordem", "energia cinética média por jogador (temperatura)")
    ke1 = rolling_mean(ke, FPS)
    dke = np.full_like(ke1, np.nan)
    dke[FPS:] = ke1[FPS:] - ke1[:-FPS]
    bag.add("kinetic_energy_d1s", dke, "m2/s2", "ordem",
            "variação da energia cinética no último 1 s (só passado)")

    print("  correlação entre vizinhos (r = {} m) ...".format(NEIGHBOR_R))
    n = len(players)
    corr = np.full(n, np.nan, dtype=np.float32)
    for lo in range(0, n, CHUNK):
        hi = min(n, lo + CHUNK)
        p, v = players[lo:hi], vel[lo:hi]
        ok = ~np.isnan(v[..., 0]) & ~np.isnan(p[..., 0])
        vm = np.nanmean(np.where(ok[..., None], v, np.nan), axis=1, keepdims=True)
        dv = np.where(ok[..., None], v - vm, 0.0)
        pp = np.where(ok[..., None], p, 1e6)
        d = np.hypot(pp[:, :, None, 0] - pp[:, None, :, 0], pp[:, :, None, 1] - pp[:, None, :, 1])
        near = (d < NEIGHBOR_R) & ok[:, :, None] & ok[:, None, :]
        near &= ~np.eye(p.shape[1], dtype=bool)[None]
        dot = np.einsum("fid,fjd->fij", dv, dv)
        mag = np.hypot(dv[..., 0], dv[..., 1])
        den = np.sum(near * (mag[:, :, None] * mag[:, None, :]), axis=(1, 2))
        num = np.sum(near * dot, axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[lo:hi] = np.where(den > 0, num / den, np.nan)
    bag.add("neighbor_align_r10", corr, "-1..1", "ordem",
            "correlação das flutuações de velocidade entre vizinhos a < {} m (Cavagna)".format(NEIGHBOR_R))


def mobility_features(bag, players, slot_teams, vel, X, index):
    print("  mobilidade ...")
    msd = {}
    for lag_s in MSD_LAGS_S:
        L = int(round(lag_s * FPS))
        d = np.full(len(players), np.nan, dtype=np.float32)
        disp = players[L:] - players[:-L]
        sq = disp[..., 0] ** 2 + disp[..., 1] ** 2
        sq[sq > (MAX_PLAYER_SPEED * lag_s) ** 2] = np.nan     # troca de identidade
        d[L:] = np.nanmean(sq, axis=1)
        msd[lag_s] = d
        bag.add("msd_{}s".format(str(lag_s).replace(".", "p")), d, "m2", "mobilidade",
                "deslocamento quadrático médio em {} s (passado)".format(lag_s))
    lt = np.log(np.array(MSD_LAGS_S))
    lm = np.log(np.stack([np.maximum(msd[l], 1e-4) for l in MSD_LAGS_S], 1))
    lt_c = lt - lt.mean()
    alpha = (lm - lm.mean(1, keepdims=True)) @ lt_c / np.sum(lt_c ** 2)
    alpha[np.isnan(lm).any(1)] = np.nan
    bag.add("msd_alpha", rolling_mean(alpha.astype(np.float32), FPS), "-", "mobilidade",
            "expoente α do MSD (0,2-2 s), média de 1 s: 2 balístico, 1 difusivo, <1 confinado")

    sp1 = X[:, index["all_speed_mean_1s"]]
    still, _ = run_lengths(sp1 < STILL_SPEED)
    bag.add("players_still_s", still / FPS, "s", "mobilidade",
            "há quanto tempo a velocidade média (1 s) está < {} m/s".format(STILL_SPEED))

    for prefix, mask in (("h", slot_teams == "H"), ("a", slot_teams == "A")):
        p, v = players[:, mask], vel[:, mask]
        cx = np.nanmean(p[..., 0], 1, keepdims=True)
        cy = np.nanmean(p[..., 1], 1, keepdims=True)
        rx, ry = p[..., 0] - cx, p[..., 1] - cy
        r = np.hypot(rx, ry)
        with np.errstate(invalid="ignore", divide="ignore"):
            rad = (v[..., 0] * rx + v[..., 1] * ry) / r
        bag.add("radial_vel_{}".format(prefix), rolling_mean(np.nanmean(rad, 1), FPS), "m/s",
                "mobilidade", "velocidade radial média (1 s) em relação ao centróide: >0 expande ({})".format(prefix))

    for name in ("all_stretch", "all_length", "all_width"):
        s = rolling_mean(X[:, index[name]], FPS)
        d = np.full_like(s, np.nan)
        d[FPS:] = s[FPS:] - s[:-FPS]
        bag.add("d_{}_1s".format(name.replace("all_", "")), d, "m/s", "mobilidade",
                "variação de {} no último 1 s".format(name))


def past_context(bag, X, index):
    print("  contexto passado ...")
    have = {name: v for name, v, *_ in bag.items}
    sources = {
        "speed": X[:, index["all_speed_mean"]],
        "pol": X[:, index["all_polarization"]],
        "chase": have["chase_cos_k3_all"],
        "ke": have["kinetic_energy"],
    }
    for key, x in sources.items():
        for w_s in PAST_WINDOWS_S:
            w = w_s * FPS
            mean = past_rolling(x, w, "mean")
            bag.add("{}_past{}s_mean".format(key, w_s), mean, "-", "passado",
                    "{}: média dos últimos {} s".format(key, w_s))
            bag.add("{}_past{}s_max".format(key, w_s), past_rolling(x, w, "max"), "-", "passado",
                    "{}: máximo dos últimos {} s".format(key, w_s))
            bag.add("{}_past{}s_trend".format(key, w_s), x - mean, "-", "passado",
                    "{}: valor atual menos a média dos últimos {} s".format(key, w_s))


# ---- I/O -----------------------------------------------------------------------------------------

def write(bag, source="tactical"):
    n = bag.n
    graded = np.zeros(n, bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    M = np.stack([v for _, v, *_ in bag.items], 1).astype(np.float32)
    M.tofile(os.path.join(OUT_DIR, "features_coletivas_{}.bin".format(source)))
    meta = {"game": GAME, "source": source, "fps": FPS, "n_frames": n, "n_features": len(bag.items),
            "features": [dict(name=a, unit=u, group=g, description=d, **describe(v, graded))
                         for a, v, u, g, d in bag.items]}
    with open(os.path.join(OUT_DIR, "features_coletivas_{}.json".format(source)), "w",
              encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1, ensure_ascii=False)
    return M, meta


def load_coletivas(source="tactical"):
    with open(os.path.join(OUT_DIR, "features_coletivas_{}.json".format(source)), encoding="utf-8") as h:
        meta = json.load(h)
    M = np.fromfile(os.path.join(OUT_DIR, "features_coletivas_{}.bin".format(source)), dtype=np.float32)
    M = M.reshape(meta["n_frames"], meta["n_features"])
    meta["index"] = {f["name"]: i for i, f in enumerate(meta["features"])}
    return M, meta


def load_all(source="tactical"):
    """X de viewer/features.py e as coletivas lado a lado, com um índice único."""
    X, m = load_features(source)
    C, c = load_coletivas(source)
    index = dict(m["index"])
    for name, i in c["index"].items():
        index[name] = X.shape[1] + i
    feats = m["features"] + c["features"]
    return np.hstack([X, C]), {"index": index, "features": feats, "n_frames": m["n_frames"]}


def main():
    t0 = time.time()
    print("carregando ATD e features ...")
    players, ball, slot_teams, _ = load_atd()
    X, meta = load_features()
    vel = centred_velocity(players, VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    bag = Bag(len(players))
    chase_features(bag, players, vel, slot_teams, ball)
    order_features(bag, players, vel, X, meta["index"])
    mobility_features(bag, players, slot_teams, vel, X, meta["index"])
    past_context(bag, X, meta["index"])
    M, out = write(bag)
    print("\n{} features x {} frames em {:.0f} s -> {}".format(M.shape[1], M.shape[0], time.time() - t0, OUT_DIR))
    for f in out["features"]:
        print("  {:<24} {:>6}  p05 {:>8.3f}  p50 {:>8.3f}  p95 {:>8.3f}  missing {:5.1%}".format(
            f["name"], f["unit"], f.get("p05", np.nan), f.get("p50", np.nan), f.get("p95", np.nan),
            f["missing"]))


if __name__ == "__main__":
    main()
