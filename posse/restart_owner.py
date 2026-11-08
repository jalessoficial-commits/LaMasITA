"""Quem começa com a bola em cada trecho de bola viva.

Todo reinício tem um dono claro (CHALLENGE.md). Este script encontra os trechos vivos a partir do
arquivo de bola morta, olha a janela em volta de cada reinício e decide o time que reinicia.

    python posse/restart_owner.py --dead <bola_morta.csv> [--ball <bola.csv>]
    python posse/restart_owner.py --dead <bola_morta.csv> --ball <bola.csv> \
        --write-submission evaluation/results/reinicio_constante.csv

Evidência usada, em ordem de confiança:

  1. Proximidade. Em cada frame da janela [s - PRE, s + POST), o time do jogador mais perto da bola
     ganha um voto, desde que esteja a menos de R_MAX metros e com folga de MARGIN metros sobre o
     adversário mais próximo. Bola disputada não vota. No reinício a bola está parada ao lado do
     cobrador, então esse sinal é forte. Onde a bola some, usa a última posição vista há no máximo
     LOOKBACK frames, porque bola parada não sai do lugar.
  2. Smart Tagging. Set pieces com Team cujo intervalo contém o reinício (com tolerância). Laterais
     vêm com Team = N/A e não votam.

O tracking decide quando é confiante; o Smart Tagging decide quando o tracking é fraco ou mudo.

Saídas:
  - posse/out/restart_owner.csv, um reinício por linha. Fica fora do git (derivado dos dados).
  - Resumo no terminal, incluindo a avaliação contra os 5 minutos públicos.
  - Opcional: uma submissão que pinta cada trecho vivo inteiro com o dono do reinício. É a linha de
    base "nenhuma troca de posse": mede o quanto só o reinício já rende no spellF1.

Formatos de entrada (ajustáveis por argumento):
  --dead   CSV frame,label com 199319 linhas. Frame com label D é bola morta; o resto é vivo.
  --ball   CSV com frame e posição da bola. Colunas em --ball-cols (padrão frame,x,y), unidades em
           --ball-units (norm = 0-1 como o ATD, m = metros), base do frame em --ball-frame-base
           (0 = frame do vídeo, 1 = frame_id do ATD). Sem --ball, usa a bola do ATD.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
PRE = 50            # frames antes do reinício (2 s): o cobrador já está junto da bola parada
POST = 6            # frames depois: logo após o toque a bola já vai para outro jogador
POST_FALLBACK = 50  # janela estendida quando a curta não tem nenhum voto
LOOKBACK = 125      # até 5 s reaproveitando a última posição vista da bola
R_MAX = 5.0         # metros: mais longe que isso, ninguém "está na bola"
MARGIN = 1.0        # metros de folga sobre o adversário mais próximo
MIN_VOTES = 5       # menos votos que isso = tracking mudo
CONF_MIN = 0.5      # |vH - vA| / (vH + vA) abaixo disso = tracking fraco
TAG_TOL = 75        # frames de tolerância em volta do intervalo do Smart Tagging (3 s)
EVAL_LEN = 25       # frames do GT olhados no início de cada trecho (1 s)

HOME, AWAY = 0, 1
LETTER = {HOME: "H", AWAY: "A", None: "X"}
OUT_DIR = os.path.join(HERE, "out")


# ------------------------------------------------------------------------------------------------
# Leitura
# ------------------------------------------------------------------------------------------------

def read_labels(path, n_frames):
    """Coluna label de um CSV frame,label, com as mesmas exigências do validador."""
    table = pd.read_csv(path)
    table.columns = [c.strip().lower() for c in table.columns]
    if len(table) != n_frames:
        raise ValueError("{}: {} linhas, esperava {}".format(path, len(table), n_frames))
    if not np.array_equal(table["frame"].values, np.arange(n_frames)):
        raise ValueError("{}: coluna frame não é 0..{}".format(path, n_frames - 1))
    return table["label"].astype(str).str.strip().str.upper().values


def live_segments(dead, blocks):
    """[(metade, início, fim_exclusivo)] dos trechos vivos dentro das metades avaliadas."""
    segments = []
    for half, first, last in blocks:
        live = ~dead[first:last + 1]
        edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            segments.append((half, first + int(start), first + int(end)))
    return segments


def team_of_each_track(meta_path):
    """0 para slot do mandante, 1 para visitante. O primeiro time do XML é o mandante."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: esperava 2 times, achei {}".format(meta_path, len(team_ids)))
    return np.array([HOME if p.get("teamId") == team_ids[0] else AWAY
                     for p in root.iter("Player")], dtype=np.int8)


def _xy(group):
    """'x,y,velocidade' -> (x, y), NaN se faltar."""
    values = group.split(",")
    try:
        return float(values[0]), float(values[1])
    except (ValueError, IndexError):
        return np.nan, np.nan


def read_atd_frames(raw_path, wanted, n_slots, n_frames):
    """Jogadores (n_frames, n_slots, 2) só nos frames pedidos, e a bola do ATD em todos.

    Uma passada pelo arquivo; só as linhas pedidas têm os jogadores decodificados, o que deixa a
    leitura bem mais rápida do que decodificar os 199319 frames.
    """
    players = {}
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id começa em 1, o vídeo em 0
            if not 0 <= frame < n_frames:
                continue
            ball[frame] = _xy(parts[2])
            if frame in wanted:
                groups = parts[1].split(";")
                pos = np.full((n_slots, 2), np.nan)
                for slot, group in enumerate(groups[:n_slots]):
                    pos[slot] = _xy(group)
                players[frame] = pos
    return players, ball


def read_ball(path, n_frames, cols, frame_base):
    """Bola de um CSV próprio, indexada por frame do vídeo. NaN onde não há bola."""
    frame_col, x_col, y_col = cols
    table = pd.read_csv(path)
    frames = table[frame_col].values.astype(int) - frame_base
    ok = (frames >= 0) & (frames < n_frames)
    ball = np.full((n_frames, 2), np.nan)
    ball[frames[ok], 0] = table[x_col].values[ok]
    ball[frames[ok], 1] = table[y_col].values[ok]
    return ball


def read_set_pieces(path, fps):
    """[(início, fim, dono, tipo)] dos SET PIECES com Team de verdade. N/A fica de fora."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        if (instance.findtext("code") or "").strip().lower() != "set pieces":
            continue
        tags = {label.findtext("group"): (label.findtext("text") or "")
                for label in instance.findall("label")}
        team = tags.get("Team", "").strip().lower()
        owner = HOME if "barcel" in team else AWAY if "atl" in team else None
        # Em "Kick Off (After Goal)" não se sabe se Team é quem sai jogando ou quem marcou; um voto
        # invertido só atrapalha, e o tracking resolve saída de bola com folga.
        if owner is None or "after goal" in tags.get("Type", "").lower():
            continue
        out.append((int(round(float(instance.findtext("start")) * fps)),
                    int(round(float(instance.findtext("end")) * fps)),
                    owner, tags.get("Type", "")))
    return out


# ------------------------------------------------------------------------------------------------
# Decisão
# ------------------------------------------------------------------------------------------------

def hold_ball(ball, lookback):
    """Repete a última posição vista da bola por até `lookback` frames."""
    n = len(ball)
    seen = np.where(~np.isnan(ball[:, 0]), np.arange(n), -1)
    last = np.maximum.accumulate(seen)
    ok = (last >= 0) & (np.arange(n) - last <= lookback)
    held = np.full_like(ball, np.nan)
    held[ok] = ball[last[ok]]
    return held


def proximity_votes(frames, players, team, ball, pitch):
    """(votos H, votos A, frames com bola) na janela."""
    votes = [0, 0]
    with_ball = 0
    scale = np.asarray(pitch, dtype=float)
    for frame in frames:
        if frame not in players or np.isnan(ball[frame, 0]):
            continue
        with_ball += 1
        distance = np.hypot(*((players[frame] - ball[frame]) * scale).T)
        tm = team[frame] if np.ndim(team) == 2 else team  # time por frame (correção do goleiro)
        nearest = [np.nanmin(distance[tm == side]) if np.any(~np.isnan(distance[tm == side]))
                   else np.inf for side in (HOME, AWAY)]
        winner = int(nearest[AWAY] < nearest[HOME])
        if nearest[winner] > R_MAX or nearest[1 - winner] - nearest[winner] < MARGIN:
            continue
        votes[winner] += 1
    return votes[HOME], votes[AWAY], with_ball


def tag_owner(start, set_pieces):
    """Dono pelo Smart Tagging: o set piece mais próximo cujo intervalo contém o reinício."""
    candidates = [(abs(first - start), owner, kind) for first, last, owner, kind in set_pieces
                  if first - TAG_TOL <= start <= last + TAG_TOL]
    if not candidates:
        return None, ""
    _, owner, kind = min(candidates)
    return owner, kind


def decide(start, end, players, team, ball, pitch, set_pieces):
    """Um reinício: dono, origem da decisão e a evidência por trás dela."""
    window = range(max(0, start - PRE), min(end, start + POST))
    v_home, v_away, with_ball = proximity_votes(window, players, team, ball, pitch)
    if v_home + v_away == 0:  # nada na janela curta: estende para dentro do trecho
        window = range(start, min(end, start + POST_FALLBACK))
        v_home, v_away, with_ball = proximity_votes(window, players, team, ball, pitch)

    total = v_home + v_away
    tracked = (HOME if v_home >= v_away else AWAY) if total >= MIN_VOTES else None
    confidence = abs(v_home - v_away) / total if total else 0.0
    tagged, kind = tag_owner(start, set_pieces)

    if tracked is not None and (confidence >= CONF_MIN or tagged is None):
        owner, source = tracked, "tracking" if confidence >= CONF_MIN else "tracking_fraco"
    elif tagged is not None:
        owner, source = tagged, "tagging"
    else:
        owner, source = None, "nenhum"

    return {"owner": LETTER[owner], "source": source, "votes_H": v_home, "votes_A": v_away,
            "frames_com_bola": with_ball, "confianca": round(confidence, 3),
            "tracking": LETTER[tracked], "tagging": LETTER[tagged], "tipo_tag": kind}


# ------------------------------------------------------------------------------------------------
# Avaliação e submissão
# ------------------------------------------------------------------------------------------------

def evaluate_public(table, gt):
    """Compara o dono do reinício com o GT público no primeiro segundo de cada trecho."""
    gt_live = np.flatnonzero(np.isin(gt, ["H", "A"]))
    gt_starts = gt_live[np.concatenate(([True], np.diff(gt_live) > 1))]
    rows = []
    for row in table.itertuples():
        if gt[row.inicio] == "X" and gt[min(row.inicio + EVAL_LEN, len(gt) - 1)] == "X":
            continue  # fora da janela pública
        piece = gt[row.inicio:row.inicio + EVAL_LEN]
        piece = piece[np.isin(piece, ["H", "A"])]
        truth = pd.Series(piece).mode().iloc[0] if len(piece) else "-"
        offset = int(gt_starts[np.argmin(np.abs(gt_starts - row.inicio))] - row.inicio) \
            if len(gt_starts) else None
        rows.append({"inicio": row.inicio, "previsto": row.owner, "gt": truth,
                     "fonte": row.source, "desvio_inicio_gt": offset})
    return pd.DataFrame(rows)


def segment_fill_submission(dead, table, blocks, n_frames):
    """D na bola morta, dono do reinício em todo o trecho vivo, X fora das metades."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = np.where(dead[first:last + 1], "D", "H")
    for row in table.itertuples():
        # Sem dono: H, o mais comum no jogo. Conta-se quantos são no resumo; deve ser perto de zero.
        labels[row.inicio:row.fim] = row.owner if row.owner in ("H", "A") else "H"
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


# ------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", required=True, help="CSV frame,label do detector de bola morta")
    parser.add_argument("--live-ha", action="store_true",
                        help="só H e A contam como vivo (X também vira morta); para usar o GT "
                             "público como --dead e testar só a janela de 5 minutos")
    parser.add_argument("--ball", help="CSV da bola própria; sem ele, usa a bola do ATD")
    parser.add_argument("--ball-cols", default="frame,x,y")
    parser.add_argument("--ball-units", choices=("norm", "m"), default="norm")
    parser.add_argument("--ball-frame-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--no-tagging", action="store_true", help="ignora o Smart Tagging")
    parser.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt",
                                                     "BAR-ATH_possession_gt_public.csv"))
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "restart_owner.csv"))
    parser.add_argument("--write-submission", help="caminho da submissão de trecho constante")
    args = parser.parse_args()

    n_frames = source_frames("tactical", GAME)
    fps = GAMES[GAME]["fps"]
    pitch = GAMES[GAME]["pitch"]
    blocks = half_video_blocks(GAME)

    labels = read_labels(args.dead, n_frames)
    dead = ~np.isin(labels, ["H", "A"]) if args.live_ha else labels == "D"
    segments = live_segments(dead, blocks)
    print("trechos vivos: {}   duração mediana {:.1f} s   curtos (< 1 s): {}".format(
        len(segments), np.median([e - s for _, s, e in segments]) / fps,
        sum(e - s < fps for _, s, e in segments)))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    wanted = set()
    for _, start, end in segments:
        wanted.update(range(max(0, start - PRE), min(end, start + POST_FALLBACK)))
    print("lendo o ATD ({} frames de interesse) ...".format(len(wanted)))
    players, atd_ball = read_atd_frames(raw_path, wanted, len(team), n_frames)

    if args.ball:
        ball = read_ball(args.ball, n_frames, args.ball_cols.split(","), args.ball_frame_base)
        if args.ball_units == "m":
            ball = ball / np.asarray(pitch, dtype=float)
        print("bola: {}".format(os.path.basename(args.ball)))
    else:
        ball = atd_ball
        print("bola: ATD")
    ball = hold_ball(ball, LOOKBACK)

    set_pieces = [] if args.no_tagging else read_set_pieces(require(["smart_tagging"])[0], fps)

    rows = []
    for half, start, end in segments:
        row = {"metade": half, "inicio": start, "fim": end, "duracao_s": round((end - start) / fps, 2)}
        row.update(decide(start, end, players, team, ball, pitch, set_pieces))
        rows.append(row)
    table = pd.DataFrame(rows)

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    table.to_csv(args.out, index=False)

    print("\norigem da decisão")
    print(table["source"].value_counts().to_string())
    print("\ndonos: H {}  A {}  sem dono {}".format(
        (table.owner == "H").sum(), (table.owner == "A").sum(), (table.owner == "X").sum()))
    both = table[(table.tracking != "X") & (table.tagging != "X")]
    print("tracking e tagging opinam juntos em {}; discordam em {}".format(
        len(both), int((both.tracking != both.tagging).sum())))

    if os.path.isfile(args.gt):
        report = evaluate_public(table, read_labels(args.gt, n_frames))
        judged = report[report["gt"] != "-"] if len(report) else report
        print("\njanela pública: {} reinícios, {} acertos".format(
            len(judged), int((judged["previsto"] == judged["gt"]).sum()) if len(judged) else 0))
        if len(report):
            print(report.to_string(index=False))

    if args.write_submission:
        write_submission(args.write_submission, segment_fill_submission(dead, table, blocks, n_frames))
        print("\nsubmissão de trecho constante: {}".format(args.write_submission))

    print("\ntabela por reinício: {}".format(args.out))


if __name__ == "__main__":
    main()