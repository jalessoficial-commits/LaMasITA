"""Per-frame features for the ball-alive / ball-dead decision tree.

    python viewer/preprocess.py        # once, if viewer/data/atd_tactical.bin is not there yet
    python viewer/features.py          # ~1-2 min

Reads what viewer/preprocess.py wrote (the quantised ATD and the Smart Tagging JSON), the homography
CSV, and -- if it exists -- the ball of tracking_bola/export_ball.py. Writes into viewer/data/:

    features_tactical.bin       float32, (n_frames, n_features), NaN = no value on that frame
    features_tactical.json      name, unit, group, Portuguese description and percentiles of each
                                feature, in column order

Index 0 is video frame 0, like every other file in viewer/data/. Everything is in metres and
seconds. Nothing here reads the possession ground truth: features come from the ATD, the homography
and the Smart Tagging only.

The metrics follow the collective-behaviour literature where there is one: polarisation and milling
(Narizuka & Yamazaki 2017; Welch, Schaerf & Murphy 2021), stretch index and surface area (Duarte et
al. 2013; Rico-Gonzalez et al. 2022) and a ball-weighted centroid (Clemente et al. 2013). Approximate
entropy is replaced by permutation entropy, which answers the same question -- how regular is the
signal -- at a fraction of the cost.

    from viewer.features import load_features
    X, meta = load_features()          # X[frame, meta["index"]["ball_speed"]]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402

# Params
OUT_DIR = os.path.join(HERE, "data")
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
VEL_HALF_WINDOW = 6        # frames each side of the centred difference: 0.48 s for players
BALL_HALF_WINDOW = 2       # 0.16 s for the ball, which changes speed far faster
MAX_PLAYER_SPEED = 12.0    # m/s; anything faster is an identity switch, not a sprint
MAX_BALL_SPEED = 40.0      # m/s
WALK_SPEED = 1.0           # m/s; below this a player counts as standing or walking
RUN_SPEED = 4.0            # m/s; above this a player counts as running
BALL_STILL_SPEED = 0.5     # m/s
PE_DELAY = 5               # permutation entropy: samples 0.2 s apart ...
PE_WINDOW = 2 * FPS        # ... counted over 2 s
HULL_STEP = 5              # convex hull every 0.2 s, interpolated in between
PENALTY_DEPTH, PENALTY_WIDTH = 16.5, 40.32
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
PENALTY_SPOT = 11.0

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected


# ---- input --------------------------------------------------------------------------------------

def load_atd(source="tactical"):
    """(players, ball, slot_teams, meta) in metres, from viewer/preprocess.py's quantised file."""
    meta_path = os.path.join(OUT_DIR, "meta_{}.json".format(source))
    bin_path = os.path.join(OUT_DIR, "atd_{}.bin".format(source))
    if not os.path.exists(bin_path):
        raise SystemExit("{} not found -- run python viewer/preprocess.py first".format(bin_path))
    with open(meta_path) as handle:
        meta = json.load(handle)
    n_obj = len(meta["slots"]) + 1
    raw = np.fromfile(bin_path, dtype=np.uint16).reshape(meta["n_frames"], n_obj, 2)
    xy = raw.astype(np.float32) / meta["scale"]
    xy[raw == meta["missing"]] = np.nan
    xy[..., 0] *= LENGTH
    xy[..., 1] *= WIDTH
    return xy[:, :-1], xy[:, -1], np.array(meta["slot_teams"]), meta


def load_tracker_ball(data_dir, n_frames):
    """tracking_bola's fused ball in metres, or None when export_ball.py has not been run here."""
    path = os.path.join(os.path.dirname(asset_paths(GAME, data_dir)["atd_raw"]),
                        "{}_tactical_ball_tracker.csv".format(GAME))
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path, usecols=["frame", "x", "y"])
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[table["frame"].values] = table[["x", "y"]].values * np.array([LENGTH, WIDTH])
    return ball


def load_reprojection_error(data_dir, n_frames):
    table = pd.read_csv(asset_paths(GAME, data_dir)["homography"],
                        usecols=["frame", "reprojection_error"])
    out = np.full(n_frames, np.nan, dtype=np.float32)
    out[table["frame"].values] = table["reprojection_error"].values
    return out


def load_set_pieces():
    path = os.path.join(OUT_DIR, "smart_tagging.json")
    if not os.path.exists(path):
        return [], []
    with open(path) as handle:
        events = json.load(handle)
    set_pieces = [e for e in events if (e["code"] or "").lower() == "set pieces"]
    phases = [e for e in events if (e["code"] or "").lower() not in ("set pieces", "goals")]
    return set_pieces, phases


# ---- small vector helpers -----------------------------------------------------------------------

def centred_velocity(xy, half_window, max_speed):
    """Velocity (m/s) by a centred difference over 2*half_window frames. NaN at the edges, where
    either end is missing, and where the jump is too fast to be a person (or a ball)."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    k = half_window
    out[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (FPS / (2.0 * k))
    speed = np.hypot(out[..., 0], out[..., 1])
    out[speed > max_speed] = np.nan
    return out


def rolling_mean(a, window):
    """Centred mean over `window` frames, ignoring NaN. NaN only where the whole window is NaN."""
    valid = ~np.isnan(a)
    values = np.concatenate(([0.0], np.cumsum(np.where(valid, a, 0.0), dtype=np.float64)))
    counts = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(a))
    low = np.maximum(0, index - window // 2)
    high = np.minimum(len(a), index + window // 2 + 1)
    n = counts[high] - counts[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (values[high] - values[low]) / n, np.nan).astype(np.float32)


def rolling_std(a, window):
    mean = rolling_mean(a, window)
    mean_sq = rolling_mean(a * a, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0)).astype(np.float32)


def nan_quantile_rows(a, q):
    """Row-wise quantile ignoring NaN, by sorting once. np.nanpercentile is ~20x slower here."""
    s = np.sort(a, axis=1)  # NaN sorts last
    n = np.sum(~np.isnan(a), axis=1)
    idx = np.clip(np.floor(q * (n - 1)).astype(int), 0, a.shape[1] - 1)
    out = s[np.arange(len(a)), idx]
    out[n == 0] = np.nan
    return out


def run_lengths(mask):
    """For each frame inside a run of True: frames since the run began, and the run's full length.
    Both are 0 outside a run."""
    n = len(mask)
    elapsed = np.zeros(n, dtype=np.float32)
    total = np.zeros(n, dtype=np.float32)
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    for start, end in zip(starts, ends):
        elapsed[start:end] = np.arange(1, end - start + 1)
        total[start:end] = end - start
    return elapsed, total


def forward_fill(a):
    """Carry the last non-NaN value forward. Frames before the first value stay NaN, because
    index 0 points at a[0], which is NaN in that case."""
    index = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(index, out=index)
    return a[index]


def permutation_entropy(signal, delay=PE_DELAY, window=PE_WINDOW):
    """Normalised permutation entropy (order 3) over a centred window. 0 = perfectly regular,
    1 = every ordinal pattern equally likely. NaN where the window has no complete pattern."""
    n = len(signal)
    a, b, c = signal[:n - 2 * delay], signal[delay:n - delay], signal[2 * delay:]
    complete = ~(np.isnan(a) | np.isnan(b) | np.isnan(c))
    # the 6 orderings of three values, as an id 0..5
    pattern = (a < b).astype(int) * 4 + (b < c).astype(int) * 2 + (a < c).astype(int)
    lookup = np.array([0, 1, -1, 2, 3, -1, 4, 5])  # 2 and 5 are impossible orderings
    pattern = lookup[pattern]
    one_hot = np.zeros((n, 6), dtype=np.float64)
    rows = np.flatnonzero(complete)
    one_hot[rows, pattern[rows]] = 1.0
    cumulative = np.vstack((np.zeros((1, 6)), np.cumsum(one_hot, axis=0)))
    index = np.arange(n)
    low = np.maximum(0, index - window // 2)
    high = np.minimum(n, index + window // 2 + 1)
    counts = cumulative[high] - cumulative[low]
    total = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = counts / total
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(6)
    h[total[:, 0] < 10] = np.nan
    return h.astype(np.float32)


def hull_areas(xy, step=HULL_STEP):
    """Convex hull area (m^2) of each frame's points; NaN with fewer than 3 points.

    One qhull call per frame is most of this script's run time, so the hull is computed every
    `step` frames and linearly interpolated in between. A team's area does not change much in
    0.2 s, and a gap longer than one step stays NaN rather than being bridged.
    """
    from scipy.spatial import ConvexHull, QhullError
    sampled = np.arange(0, len(xy), step)
    area = np.full(len(sampled), np.nan)
    for i, f in enumerate(sampled):
        points = xy[f][~np.isnan(xy[f, :, 0])]
        if len(points) >= 3:
            try:
                area[i] = ConvexHull(points).volume  # in 2-D, volume is the area
            except QhullError:
                area[i] = 0.0
    out = np.full(len(xy), np.nan, dtype=np.float32)
    ok = ~np.isnan(area)
    frames = np.arange(len(xy))
    out[:] = np.interp(frames, sampled[ok], area[ok])
    nearest = np.clip(np.round(frames / step).astype(int), 0, len(sampled) - 1)
    out[~ok[nearest]] = np.nan
    return out


# ---- feature groups -----------------------------------------------------------------------------

class Features:
    """Ordered collection of (name, values, unit, group, description)."""

    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.items = []

    def add(self, name, values, unit, group, description):
        values = np.asarray(values, dtype=np.float32)
        assert values.shape == (self.n_frames,), name
        self.items.append((name, values, unit, group, description))


def attack_sign(players, slot_teams, n_frames):
    """+1 on frames where the home team attacks towards x = LENGTH, -1 where it attacks towards 0.

    Decided once per half from where the home side stands on average: a team spends more time in
    its own half than in the other one only if it is defending it. Frames outside the halves take
    the nearest half's direction, which matters for nothing graded.
    """
    sign = np.ones(n_frames, dtype=np.float32)
    blocks = half_video_blocks(GAME)
    home_x = np.nanmean(players[:, slot_teams == "H", 0], axis=1)
    boundaries = [0] + [blocks[1][1]] + [n_frames]
    for (half, first, last), lo, hi in zip(blocks, boundaries[:-1], boundaries[1:]):
        mean_x = np.nanmean(home_x[first:last + 1])
        sign[lo:hi] = 1.0 if mean_x < LENGTH / 2 else -1.0
        print("  half {}: home defends {} (mean x {:.1f} m)".format(
            half, "x = 0" if mean_x < LENGTH / 2 else "x = {}".format(LENGTH), mean_x))
    return sign


def add_ball_features(feats, ball, players, prefix, label):
    """Ball position, motion, gaps and its relation to the lines, restart spots and players."""
    group = "bola" if prefix == "ball" else "bola_tracker"
    p = prefix + "_"
    x, y = ball[:, 0], ball[:, 1]
    present = ~np.isnan(x)
    feats.add(p + "present", present, "0/1", group, "bola visível no frame ({})".format(label))

    since, total = run_lengths(~present)
    feats.add(p + "gap_since_s", since / FPS, "s", group,
              "tempo desde que a bola sumiu (0 se visível)")
    feats.add(p + "gap_len_s", total / FPS, "s", group,
              "duração total da lacuna atual de bola (olha o futuro; 0 se visível)")

    out_dist = np.maximum.reduce([-x, x - LENGTH, -y, y - WIDTH])
    feats.add(p + "out_dist_m", out_dist, "m", group,
              "distância da bola para fora das linhas (>0 = fora do campo, <0 = dentro)")
    feats.add(p + "last_out_dist_m", forward_fill(out_dist), "m", group,
              "última distância para fora conhecida (mantida durante a lacuna)")
    feats.add(p + "touchline_d_m", np.minimum(y, WIDTH - y), "m", group,
              "distância à linha lateral mais próxima")
    feats.add(p + "goalline_d_m", np.minimum(x, LENGTH - x), "m", group,
              "distância à linha de fundo mais próxima")
    feats.add(p + "center_d_m", np.hypot(x - LENGTH / 2, y - WIDTH / 2), "m", group,
              "distância ao centro do campo (saída de bola)")
    corner = np.min([np.hypot(x - cx, y - cy) for cx in (0, LENGTH) for cy in (0, WIDTH)], axis=0)
    feats.add(p + "corner_d_m", corner, "m", group, "distância à bandeira de escanteio mais próxima")
    spot = np.minimum(np.hypot(x - PENALTY_SPOT, y - WIDTH / 2),
                      np.hypot(x - (LENGTH - PENALTY_SPOT), y - WIDTH / 2))
    feats.add(p + "penalty_spot_d_m", spot, "m", group, "distância à marca do pênalti mais próxima")
    dx_ga = np.maximum(np.minimum(x, LENGTH - x) - GOAL_AREA_DEPTH, 0)
    dy_ga = np.maximum(np.abs(y - WIDTH / 2) - GOAL_AREA_WIDTH / 2, 0)
    feats.add(p + "goal_area_d_m", np.hypot(dx_ga, dy_ga), "m", group,
              "distância à pequena área mais próxima (0 = dentro; tiro de meta)")

    vel = centred_velocity(ball, BALL_HALF_WINDOW, MAX_BALL_SPEED)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    feats.add(p + "speed", speed, "m/s", group, "velocidade da bola")
    feats.add(p + "speed_1s", rolling_mean(speed, FPS), "m/s", group,
              "velocidade média da bola em 1 s")
    still, _ = run_lengths(speed < BALL_STILL_SPEED)
    feats.add(p + "still_s", still / FPS, "s", group,
              "há quanto tempo a bola está parada (< {} m/s)".format(BALL_STILL_SPEED))

    d = np.hypot(players[..., 0] - x[:, None], players[..., 1] - y[:, None])
    feats.add(p + "nearest_player_d_m", np.nanmin(d, axis=1), "m", group,
              "distância do jogador mais próximo à bola")
    feats.add(p + "players_5m", np.where(present, np.sum(d < 5, axis=1), np.nan), "n", group,
              "nº de jogadores a menos de 5 m da bola")
    feats.add(p + "players_10m", np.where(present, np.sum(d < 10, axis=1), np.nan), "n", group,
              "nº de jogadores a menos de 10 m da bola")


def team_block(players, vel, prefix, label):
    """Kinematic and shape metrics of one group of players. Returns dict name -> (values, unit, desc)."""
    out = {}
    x, y = players[..., 0], players[..., 1]
    visible = ~np.isnan(x)
    n = visible.sum(axis=1).astype(np.float32)
    out["n"] = (n, "n", "nº de slots rastreados ({})".format(label))

    speed = np.hypot(vel[..., 0], vel[..., 1])
    has_speed = ~np.isnan(speed)
    n_speed = has_speed.sum(axis=1)
    out["speed_mean"] = (np.nanmean(speed, axis=1), "m/s", "velocidade média ({})".format(label))
    out["speed_median"] = (nan_quantile_rows(speed, 0.5), "m/s",
                           "velocidade mediana ({})".format(label))
    out["speed_p90"] = (nan_quantile_rows(speed, 0.9), "m/s",
                        "percentil 90 da velocidade ({})".format(label))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["frac_walk"] = (np.where(n_speed > 0, np.sum(speed < WALK_SPEED, axis=1) / n_speed, np.nan),
                            "0-1", "fração parada/andando, < {} m/s ({})".format(WALK_SPEED, label))
        out["frac_run"] = (np.where(n_speed > 0, np.sum(speed > RUN_SPEED, axis=1) / n_speed, np.nan),
                           "0-1", "fração correndo, > {} m/s ({})".format(RUN_SPEED, label))

        # polarisation: length of the mean unit velocity (1 = all run the same way)
        moving = speed > 0.3
        ux = np.where(moving, vel[..., 0] / speed, np.nan)
        uy = np.where(moving, vel[..., 1] / speed, np.nan)
        out["polarization"] = (np.hypot(np.nanmean(ux, axis=1), np.nanmean(uy, axis=1)), "0-1",
                               "polarização: alinhamento das direções de corrida ({})".format(label))

        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        rx, ry = x - cx[:, None], y - cy[:, None]
        r = np.hypot(rx, ry)
        cross = (rx / r) * uy - (ry / r) * ux
        out["milling"] = (np.abs(np.nanmean(cross, axis=1)), "0-1",
                          "milling: rotação coletiva em torno do centróide ({})".format(label))

    cvel = centred_velocity(np.stack([cx, cy], axis=1), VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    out["centroid_speed"] = (np.hypot(cvel[:, 0], cvel[:, 1]), "m/s",
                             "velocidade do centróide ({})".format(label))
    out["_cvel"] = cvel  # kept for the inter-team metrics, not written
    out["_centroid"] = np.stack([cx, cy], axis=1)
    out["stretch"] = (np.nanmean(r, axis=1), "m",
                      "stretch index: distância média ao centróide ({})".format(label))
    out["length"] = (np.nanmax(x, axis=1) - np.nanmin(x, axis=1), "m",
                     "comprimento (eixo do campo) ({})".format(label))
    out["width"] = (np.nanmax(y, axis=1) - np.nanmin(y, axis=1), "m",
                    "largura ({})".format(label))
    return out


def add_player_features(feats, players, slot_teams, ball, sign):
    vel = centred_velocity(players, VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    blocks = {}
    for prefix, mask, label in (("all", np.ones(len(slot_teams), bool), "todos"),
                                ("h", slot_teams == "H", "casa"),
                                ("a", slot_teams == "A", "fora")):
        print("  players: {} ...".format(label))
        block = team_block(players[:, mask], vel[:, mask], prefix, label)
        blocks[prefix] = block
        group = "jogadores" if prefix == "all" else "time"
        for name, item in block.items():
            if not name.startswith("_"):
                feats.add("{}_{}".format(prefix, name), item[0], item[1], group, item[2])
        print("  players: {} hull ...".format(label))
        feats.add(prefix + "_hull_area", hull_areas(players[:, mask]), "m2", group,
                  "área do fecho convexo, surface area ({})".format(label))

    # temporal context of the collective speed
    speed = blocks["all"]["speed_mean"][0]
    feats.add("all_speed_mean_1s", rolling_mean(speed, FPS), "m/s", "jogadores",
              "velocidade média de todos, média móvel de 1 s")
    feats.add("all_speed_mean_3s", rolling_mean(speed, 3 * FPS), "m/s", "jogadores",
              "velocidade média de todos, média móvel de 3 s")
    smooth = rolling_mean(speed, FPS)
    accel = np.full_like(smooth, np.nan)
    accel[FPS // 2:FPS // 2 + len(smooth) - FPS] = smooth[FPS:] - smooth[:-FPS]  # per 1 s
    feats.add("all_speed_accel", accel, "m/s2", "jogadores",
              "variação da velocidade média de todos em 1 s (aceleração coletiva)")
    feats.add("all_speed_std_2s", rolling_std(speed, 2 * FPS), "m/s", "jogadores",
              "desvio-padrão da velocidade média de todos em 2 s")
    feats.add("all_centroid_pe_2s", permutation_entropy(blocks["all"]["centroid_speed"][0]), "0-1",
              "jogadores", "entropia de permutação da velocidade do centróide em 2 s (regularidade)")

    # between the teams
    ch, ca = blocks["h"]["_centroid"], blocks["a"]["_centroid"]
    feats.add("teams_centroid_d_m", np.hypot(*(ch - ca).T), "m", "time",
              "distância entre os centróides dos dois times")
    vh, va = blocks["h"]["_cvel"], blocks["a"]["_cvel"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(vh * va, axis=1) / (np.hypot(*vh.T) * np.hypot(*va.T))
    feats.add("teams_centroid_vel_cos", cos, "-1..1", "time",
              "cosseno entre as velocidades dos centróides (times se movendo juntos)")

    # own half: x measured from each team's own goal line
    home_attacks_up = sign[:, None] > 0
    x = players[..., 0]
    own_x = np.where((slot_teams == "H")[None, :] == home_attacks_up, x, LENGTH - x)
    visible = ~np.isnan(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats.add("all_frac_own_half", np.sum(own_x < LENGTH / 2, axis=1) / visible.sum(axis=1), "0-1",
                  "jogadores", "fração de jogadores no próprio campo (1 = formação de saída de bola)")
    in_box = ((np.minimum(x, LENGTH - x) < PENALTY_DEPTH)
              & (np.abs(players[..., 1] - WIDTH / 2) < PENALTY_WIDTH / 2))
    feats.add("all_players_in_boxes", np.sum(in_box, axis=1), "n", "jogadores",
              "nº de jogadores dentro das grandes áreas (escanteio, falta)")
    h_x = blocks["h"]["_centroid"][:, 0]
    feats.add("h_centroid_x_att", np.where(sign > 0, h_x, LENGTH - h_x), "m", "time",
              "centróide da casa no eixo de ataque (0 = própria linha de fundo)")
    a_x = blocks["a"]["_centroid"][:, 0]
    feats.add("a_centroid_x_att", np.where(sign > 0, LENGTH - a_x, a_x), "m", "time",
              "centróide de fora no eixo de ataque (0 = própria linha de fundo)")

    # ball-weighted centroid (Clemente): players near the ball count more
    bx, by = ball[:, 0:1], ball[:, 1:2]
    d = np.hypot(players[..., 0] - bx, players[..., 1] - by)
    w = 1.0 / (1.0 + d)
    with np.errstate(invalid="ignore", divide="ignore"):
        wx = np.nansum(w * players[..., 0], axis=1) / np.nansum(np.where(visible, w, 0), axis=1)
        wy = np.nansum(w * players[..., 1], axis=1) / np.nansum(np.where(visible, w, 0), axis=1)
    feats.add("ball_wcentroid_d_m", np.where(np.isnan(ball[:, 0]), np.nan, np.hypot(wx - ball[:, 0], wy - ball[:, 1])),
              "m", "bola", "distância da bola ao centróide ponderado pela proximidade (wCentroid)")


def add_context_features(feats, n_frames, reprojection, set_pieces, phases):
    group = "contexto"
    in_half = np.zeros(n_frames, dtype=np.float32)
    half_t = np.full(n_frames, np.nan, dtype=np.float32)
    for half, first, last in half_video_blocks(GAME):
        in_half[first:last + 1] = half
        half_t[first:last + 1] = np.arange(last - first + 1) / FPS
    feats.add("half", in_half, "0/1/2", group, "tempo de jogo avaliado (0 = fora dos tempos)")
    feats.add("half_time_s", half_t, "s", group, "segundos desde o início do tempo")
    feats.add("reproj_error", reprojection, "px", group, "erro de reprojeção da homografia")

    sp_active = np.zeros(n_frames, dtype=np.float32)
    starts = np.array(sorted(e["start_frame"] for e in set_pieces), dtype=np.int64)
    for e in set_pieces:
        sp_active[max(e["start_frame"], 0):e["end_frame"]] = 1
    feats.add("st_set_piece", sp_active, "0/1", group, "dentro de um SET PIECE do Smart Tagging")
    frames = np.arange(n_frames)
    to_next = np.full(n_frames, np.nan, dtype=np.float32)
    since = np.full(n_frames, np.nan, dtype=np.float32)
    if len(starts):
        k = np.searchsorted(starts, frames, side="left")
        ok = k < len(starts)
        to_next[ok] = (starts[k[ok]] - frames[ok]) / FPS
        ok = k > 0
        since[ok] = (frames[ok] - starts[k[ok] - 1]) / FPS
    feats.add("st_to_next_sp_s", to_next, "s", group, "segundos até o próximo SET PIECE (reinício)")
    feats.add("st_since_sp_s", since, "s", group, "segundos desde o último SET PIECE")

    phase_active = np.zeros(n_frames, dtype=np.float32)
    for e in phases:
        phase_active[max(e["start_frame"], 0):e["end_frame"]] = 1
    feats.add("st_phase", phase_active, "0/1", group,
              "dentro de alguma fase de jogo do Smart Tagging (build up, progression...)")


# ---- output -------------------------------------------------------------------------------------

def describe(values, graded):
    v = values[graded]
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"missing": 1.0}
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {"missing": float(1 - len(v) / graded.sum()),
            "min": float(q[0]), "p01": float(q[1]), "p05": float(q[2]), "p25": float(q[3]),
            "p50": float(q[4]), "p75": float(q[5]), "p95": float(q[6]), "p99": float(q[7]),
            "max": float(q[8])}


def write(feats, source="tactical"):
    graded = np.zeros(feats.n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    matrix = np.stack([values for _, values, _, _, _ in feats.items], axis=1).astype(np.float32)
    matrix.tofile(os.path.join(OUT_DIR, "features_{}.bin".format(source)))
    meta = {
        "game": GAME, "source": source, "fps": FPS, "pitch": [LENGTH, WIDTH],
        "n_frames": feats.n_frames, "n_features": len(feats.items),
        "features": [dict(name=name, unit=unit, group=group, description=desc,
                          **describe(values, graded))
                     for name, values, unit, group, desc in feats.items],
    }
    with open(os.path.join(OUT_DIR, "features_{}.json".format(source)), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1, ensure_ascii=False)
    return matrix, meta


def load_features(source="tactical"):
    """(matrix (n_frames, n_features) float32, meta) with meta["index"] = {name: column}."""
    with open(os.path.join(OUT_DIR, "features_{}.json".format(source)), encoding="utf-8") as handle:
        meta = json.load(handle)
    matrix = np.fromfile(os.path.join(OUT_DIR, "features_{}.bin".format(source)), dtype=np.float32)
    matrix = matrix.reshape(meta["n_frames"], meta["n_features"])
    meta["index"] = {f["name"]: i for i, f in enumerate(meta["features"])}
    return matrix, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    print("loading ATD ...")
    players, ball, slot_teams, meta = load_atd()
    n_frames = meta["n_frames"]
    feats = Features(n_frames)
    sign = attack_sign(players, slot_teams, n_frames)

    print("ball (ATD) ...")
    add_ball_features(feats, ball, players, "ball", "ATD")
    tracker = load_tracker_ball(args.data_dir, n_frames)
    if tracker is not None:
        print("ball (tracking_bola) ...")
        add_ball_features(feats, tracker, players, "trk", "tracking_bola")
    else:
        print("ball (tracking_bola): BAR-ATH_tactical_ball_tracker.csv not found, skipped")

    print("players ...")
    add_player_features(feats, players, slot_teams, ball, sign)

    print("context ...")
    set_pieces, phases = load_set_pieces()
    add_context_features(feats, n_frames, load_reprojection_error(args.data_dir, n_frames),
                         set_pieces, phases)

    matrix, out_meta = write(feats)
    print("\n{} features x {} frames -> {}".format(matrix.shape[1], matrix.shape[0], OUT_DIR))
    for f in out_meta["features"]:
        print("  {:<26} {:>6}  p05 {:>9.3f}  p50 {:>9.3f}  p95 {:>9.3f}  missing {:5.1%}".format(
            f["name"], f["unit"], f.get("p05", np.nan), f.get("p50", np.nan), f.get("p95", np.nan),
            f["missing"]))


if __name__ == "__main__":
    main()
