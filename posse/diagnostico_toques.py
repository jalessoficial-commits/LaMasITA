"""Diagnóstico: o que distingue um toque que o GT conta como troca de posse de um que ele ignora?

    python posse/diagnostico_toques.py --ball data\\tactical\\BAR-ATH_tactical_ball_tracker.csv
    python posse/diagnostico_toques.py --ball tracking_bola/out/viterbi_bola.csv --ball-units px

Usa só os 5 minutos públicos: os trechos vivos vêm do próprio GT (H e A), como no --live-ha.

1. Acha os EVENTOS da bola: picos locais da mudança do vetor velocidade (média dos 4 frames antes
   contra a dos 4 depois), com um limiar baixo de propósito (2x o ruído típico), para não perder
   nenhum candidato. Para cada evento mede:
     ângulo   entre a direção antes e a depois, em graus (só se a bola anda a pelo menos
              --min-speed m/s dos dois lados; senão a direção é ruído e fica "-")
     dv       |v_depois - v_antes|, m/s
     v1, v2   velocidade antes e depois, m/s (recepção: v1 alto, v2 baixo; chute: o contrário)
     d        distância da bola ao jogador mais perto de cada time, m
2. Grupo TROCA: para cada troca de posse do GT (H <-> A com a bola viva), o evento a até 1 s dela
   em que o time mais perto da bola é o NOVO time. Se não houver, a troca aparece como "sem evento".
3. Grupo IGNORADO: eventos em que o time mais perto da bola é o que NÃO tem a posse segundo o GT,
   a mais de 1 s de qualquer troca do GT. São os toques do adversário que o GT não contou.
4. Tabela de separação: para cada limiar de ângulo e de dv, quantos de cada grupo ficam acima. Um
   bom critério pega quase todo o grupo TROCA e quase nada do IGNORADO.

São 12 trocas: a tabela serve para ver SE os grupos se separam e em que faixa, não para cravar um
número com precisão. Confira os casos limítrofes no viewer antes de escolher o limiar.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import restart_owner as ro  # noqa: E402
import predictor as pr  # noqa: E402

# Params
EVENT_NOISE_FACTOR = 2.0  # limiar dos eventos: 2x a mediana de dv (o preditor usa 3x)
MATCH_WINDOW = 25         # frames (1 s) entre uma troca do GT e o evento que a explica
NEAR_MAX = 4.0            # m: evento com ninguém a menos disso não é atribuído a time nenhum
ANGLES = (15, 30, 45, 60, 90)
DVS = (4, 6, 8, 10, 14)


def velocities(ball_m, segments, fps):
    """(v_antes, v_depois) por frame, médias de VEL_SPAN frames de cada lado; NaN sem dados."""
    velocity = np.full_like(ball_m, np.nan)
    velocity[1:-1] = (ball_m[2:] - ball_m[:-2]) * fps / 2.0
    v_in = np.full_like(ball_m, np.nan)
    v_out = np.full_like(ball_m, np.nan)
    for _, start, end in segments:
        for t in range(start + 1, end - 1):
            v_in[t] = pr.nanmean_window(velocity, t - pr.VEL_SPAN, t)
            v_out[t] = pr.nanmean_window(velocity, t + 1, t + 1 + pr.VEL_SPAN)
    return v_in, v_out


def angle_deg(a, b, min_speed):
    """Ângulo entre dois vetores, em graus; NaN se algum for mais lento que min_speed."""
    na, nb = np.hypot(*a), np.hypot(*b)
    if not (na >= min_speed and nb >= min_speed):
        return np.nan
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def find_events(dv, segments, threshold):
    """Frames de pico local de dv acima do limiar, dentro dos trechos vivos."""
    events = []
    for _, start, end in segments:
        piece = dv[start:end]
        for i in np.flatnonzero(piece >= threshold):
            if piece[i] >= np.nanmax(piece[max(0, i - 3):i + 4]):
                events.append(start + int(i))
    return events


def describe(t, v_in, v_out, d_home, d_away, min_speed):
    lo, hi = max(0, t - 2), t + 3
    dh, da = float(d_home[lo:hi].min()), float(d_away[lo:hi].min())
    return {"frame": t, "angulo": angle_deg(v_in[t], v_out[t], min_speed),
            "dv": float(np.hypot(*(v_out[t] - v_in[t]))),
            "v1": float(np.hypot(*v_in[t])), "v2": float(np.hypot(*v_out[t])),
            "dH": dh, "dA": da, "perto": "H" if dh < da else "A", "dmin": min(dh, da)}


def fmt(x, digits=0):
    return "-" if x is None or not np.isfinite(x) else "{:.{}f}".format(x, digits)


def print_rows(title, rows, extra=None):
    print("\n" + title)
    print("  {:>7} {:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5}{}".format(
        "frame", "ang°", "dv", "v1", "v2", "dH", "dA", "perto", "  " + extra[0] if extra else ""))
    for row in rows:
        if row.get("frame") is None:
            print("  {:>7}  {} sem evento do novo time a até 1 s  {}".format(
                row["gt_frame"], row.get("para", ""), row.get("nota", "")))
            for other in row.get("outros", []):
                print("  {:>7} {:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5}  <- outro evento na janela ({:+d})"
                      .format(other["frame"], fmt(other["angulo"]), fmt(other["dv"], 1),
                              fmt(other["v1"], 1), fmt(other["v2"], 1), fmt(other["dH"], 1),
                              fmt(other["dA"], 1), other["perto"], other["frame"] - row["gt_frame"]))
            continue
        print("  {:>7} {:>5} {:>6} {:>5} {:>5} {:>5} {:>5} {:>5}{}".format(
            row["frame"], fmt(row["angulo"]), fmt(row["dv"], 1), fmt(row["v1"], 1), fmt(row["v2"], 1),
            fmt(row["dH"], 1), fmt(row["dA"], 1), row["perto"],
            "  " + extra[1](row) if extra else ""))


def separation(switch, ignored):
    """Fração de cada grupo acima de cada limiar."""
    def frac(rows, key, value):
        vals = np.array([r[key] for r in rows if r.get("frame") is not None], dtype=float)
        if not len(rows):
            return "-"
        return "{}/{}".format(int(np.sum(vals >= value)), len(rows))
    print("\nSEPARAÇÃO (quantos de cada grupo ficam acima do limiar; ângulo '-' conta como abaixo)")
    print("  {:>12} {:>10} {:>10}".format("limiar", "TROCA", "IGNORADO"))
    for a in ANGLES:
        print("  {:>12} {:>10} {:>10}".format("ângulo>={}°".format(a),
                                              frac(switch, "angulo", a), frac(ignored, "angulo", a)))
    for d in DVS:
        print("  {:>12} {:>10} {:>10}".format("dv>={} m/s".format(d),
                                              frac(switch, "dv", d), frac(ignored, "dv", d)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ball", help="CSV da bola; sem ele, a do ATD")
    parser.add_argument("--ball-cols", default="frame,x,y")
    parser.add_argument("--ball-units", choices=("norm", "m", "px"), default="norm")
    parser.add_argument("--ball-frame-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--no-despike", action="store_true")
    parser.add_argument("--no-gk-fix", action="store_true", help="não corrige o time do goleiro")
    parser.add_argument("--min-speed", type=float, default=3.0,
                        help="m/s mínimos dos dois lados para medir o ângulo (padrão 3)")
    parser.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt",
                                                     "BAR-ATH_possession_gt_public.csv"))
    args = parser.parse_args()

    n_frames = source_frames("tactical", GAME)
    fps = GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=float)
    gt = ro.read_labels(args.gt, n_frames)
    segments = ro.live_segments(~np.isin(gt, ["H", "A"]), half_video_blocks(GAME))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = ro.team_of_each_track(meta_path)
    players, atd_ball = pr.read_atd_players(raw_path, len(team), n_frames)
    if args.ball:
        ball = ro.read_ball(args.ball, n_frames, args.ball_cols.split(","), args.ball_frame_base)
        ball = ball / pitch if args.ball_units == "m" else (
            pr.pixels_to_field(ball) if args.ball_units == "px" else ball)
    else:
        ball = atd_ball
    raw_m = ball * pitch
    clean_m = raw_m if args.no_despike else pr.despike(raw_m, pr.SPIKE_SPAN, pr.SPIKE_DIST)
    ball_m = pr.smooth(clean_m, pr.SMOOTH)
    teams = team if args.no_gk_fix else pr.goalkeeper_teams(  # [0]: só os times por frame
        players, team, half_video_blocks(GAME), pitch)[0]
    d_home, d_away = pr.team_distances(players * pitch.astype(np.float32), ball_m, teams)

    v_in, v_out = velocities(ball_m, segments, fps)
    dv = np.hypot(*(v_out - v_in).T)
    noise = float(np.nanmedian(dv[np.concatenate([np.arange(s, e) for _, s, e in segments])]))
    threshold = EVENT_NOISE_FACTOR * noise
    events = find_events(dv, segments, threshold)
    print("bola: {}   ruído (dv mediano) {:.1f} m/s   limiar dos eventos {:.1f} m/s   eventos: {}"
          .format(os.path.basename(args.ball) if args.ball else "ATD", noise, threshold, len(events)))

    every = [describe(t, v_in, v_out, d_home, d_away, args.min_speed) for t in events]
    described = [r for r in every if r["dmin"] <= NEAR_MAX]

    # Trocas do GT com a bola viva
    switches = [(t, gt[t]) for t in range(1, n_frames)
                if gt[t] in ("H", "A") and gt[t - 1] in ("H", "A") and gt[t] != gt[t - 1]]
    switch_rows, used = [], set()
    for f, to in switches:
        near = [r for r in described if abs(r["frame"] - f) <= MATCH_WINDOW and r["perto"] == to]
        if near:
            best = max(near, key=lambda r: r["dv"])
            used.add(best["frame"])
            switch_rows.append(dict(best, gt_frame=f, para=to, desvio=best["frame"] - f))
        else:
            missing = np.isnan(ball_m[max(0, f - MATCH_WINDOW):f + MATCH_WINDOW, 0]).mean()
            others = [r for r in every if abs(r["frame"] - f) <= MATCH_WINDOW]
            switch_rows.append({"frame": None, "gt_frame": f, "para": to, "outros": others,
                                "nota": "(bola ausente em {:.0%} da janela; {} evento(s) na "
                                        "janela, nenhum com o novo time mais perto)".format(
                                            missing, len(others))})

    switch_frames = np.array([f for f, _ in switches])
    ignored = [r for r in described
               if r["perto"] != gt[r["frame"]]
               and (not len(switch_frames) or np.abs(switch_frames - r["frame"]).min() > MATCH_WINDOW)]

    print_rows("TROCA: evento que explica cada troca do GT (time mais perto = novo time)",
               switch_rows, ("GT (desvio)", lambda r: "{} {} ({:+d})".format(
                   r["gt_frame"], r["para"], r["desvio"])))
    print_rows("IGNORADO: eventos com o adversário da posse mais perto, longe de qualquer troca do GT",
               ignored)
    separation(switch_rows, ignored)


if __name__ == "__main__":
    main()