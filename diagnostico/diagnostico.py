"""Diagnóstico do tracking (ATD): o que há nos dados e o time de cada slot é confiável?
 
Rode a partir da raiz do repositório (só numpy; reusa starter/baseline.py e games.py):
 
    python diagnostico/diagnostico.py
 
Lê o atd.txt uma vez (~1-2 min) e guarda um cache em data/_cache/ (data/ é ignorado pelo git;
não publique esse cache, é dado derivado do tracking). Imprime, só com o ATD e sem nenhum GT:
 
  0. sanidade do parse: cobertura da bola vs. números publicados em DATA.md
  1. slots ativos por time por frame graduado
  2. objetos por frame em cada segmento do vídeo (pré-jogo, intervalo, pós-jogo...)
  3. validade dos valores: x/y parciais, fora de [0,1], posição exata (0,0)
  4. saltos de posição e reaparições impossíveis (sinal de troca de identidade)
  5. duração das lacunas (jogadores e bola)
  6. duplicatas: dois slots quase na mesma posição no mesmo frame
  7. coerência espacial: o slot está mais perto do centroide do próprio time ou do outro?
  8. as coordenadas normalizadas espelham no 2º tempo?
 
Graduado = frames que o scorer avalia (1º tempo 11185-90651, 2º tempo 110808-190197). Os limiares
(--jump-m, --max-speed, --dup-m) são escolhas minhas, não vêm dos docs: olhe as distribuições.
"""
 
import argparse
import os
import sys
 
import numpy as np
 
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))  # baseline, load_homography
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import baseline  # noqa: E402  (reusa team_of_each_track)
 
# Params
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]                      # (comprimento, largura) em metros
CACHE = os.path.join(DATA_DIR, "_cache", "atd_xy_v2.npz")
HOME, AWAY = 0, 1
REAPPEAR_MARGIN_M = 2.0                           # folga de ruído ao julgar uma reaparição
KICKOFF_2ND = 113742                              # apito do 2º tempo (FAQ.md / DATA.md)
# Números publicados em DATA.md, para conferir o parse.
DOC_BALL_ALL = (113488, 199319)
DOC_BALL_GRADED = (94214, 158857)
# Faixas de duração de lacuna, em frames (25 fps): (mínimo, máximo, rótulo).
GAP_BINS = ((1, 5, "<=0.2 s"), (6, 25, "0.2-1 s"), (26, 50, "1-2 s"), (51, 125, "2-5 s"),
            (126, 250, "5-10 s"), (251, None, ">10 s"))
 
 
def parse(raw_path, n_frames, n_slots):
    """(xy, ball): xy (n_frames, n_slots, 2) e ball (n_frames, 2), normalizados 0-1, NaN = ausente.
 
    Formato da linha: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,...>" (frame_id começa em 1).
    Só o grupo "NaN,NaN..." é pulado sem ler; qualquer outro é convertido, então um x/y parcial
    ou um marcador estranho (-1, 0) chega à análise em vez de sumir.
    """
    xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    checked = False
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            groups = parts[1].split(";")
            if not checked:
                if len(groups) != n_slots:
                    raise ValueError("a linha tem %d slots mas o XML lista %d; a ordem slot->time "
                                     "não se aplica" % (len(groups), n_slots))
                checked = True
            frame = int(parts[0]) - 1
            for slot, group in enumerate(groups):
                if not group or group.startswith("NaN,NaN"):
                    continue
                values = group.split(",")
                xy[frame, slot, 0] = float(values[0])
                xy[frame, slot, 1] = float(values[1])
            text = parts[2].strip()
            if text and not text.startswith("NaN,NaN"):
                values = text.split(",")
                ball[frame, 0] = float(values[0])
                ball[frame, 1] = float(values[1])
    return xy, ball
 
 
def load(use_cache):
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = baseline.team_of_each_track(meta_path)
    n_frames = source_frames("tactical")
    if use_cache and os.path.isfile(CACHE):
        cached = np.load(CACHE)
        if cached["xy"].shape == (n_frames, len(team), 2):
            return cached["xy"], cached["ball"], team
    print("lendo %s ..." % os.path.basename(raw_path))
    xy, ball = parse(raw_path, n_frames, len(team))
    if use_cache:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, xy=xy, ball=ball)
    return xy, ball, team
 
 
def half_of_frames(n_frames):
    """0 = não graduado, 1 = 1º tempo, 2 = 2º tempo."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in half_video_blocks(GAME):
        half[first:last + 1] = number
    return half
 
 
def run_lengths(flags):
    """Comprimento de cada sequência consecutiva de True."""
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    change = np.diff(padded)
    return np.flatnonzero(change == -1) - np.flatnonzero(change == 1)
 
 
def longest_run(flags):
    runs = run_lengths(flags)
    return int(runs.max()) if len(runs) else 0
 
 
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
 
 
def check_ball(ball, graded):
    section("0. sanidade do parse: bola")
    seen = ~np.isnan(ball[:, 0])
    for name, got, doc in (("todos os frames", (int(seen.sum()), len(seen)), DOC_BALL_ALL),
                           ("frames graduados", (int(seen[graded].sum()), int(graded.sum())),
                            DOC_BALL_GRADED)):
        print("  %-17s %7d / %7d (%.1f%%)   DATA.md: %d / %d" % (
            name, got[0], got[1], 100.0 * got[0] / got[1], doc[0], doc[1]))
    print("  Se os números não baterem, o parse (ou o alinhamento) está errado: pare aqui.")
 
 
def team_counts(present, team):
    section("1. slots ativos por time, por frame graduado")
    print("  slots no XML: casa %d, fora %d" % ((team == HOME).sum(), (team == AWAY).sum()))
    for name, side in (("casa", HOME), ("fora", AWAY)):
        n = present[:, team == side].sum(axis=1)
        counts = np.bincount(n)
        shares = ", ".join("%d: %.1f%%" % (k, 100.0 * c / len(n))
                           for k, c in enumerate(counts) if c / len(n) >= 0.01)
        print("  %-5s média %.2f | >11: %.1f%% | <9: %.1f%% | %s" % (
            name, n.mean(), 100.0 * (n > 11).mean(), 100.0 * (n < 9).mean(), shares))
    total = present.sum(axis=1)
    print("  total  média %.2f | >22: %.1f%%   (DATA.md: média 20.5)" % (
        total.mean(), 100.0 * (total > 22).mean()))
 
 
def segments(xy, ball):
    section("2. objetos por frame em cada segmento do vídeo")
    (_, first1, last1), (_, first2, last2) = half_video_blocks(GAME)
    n_frames = len(xy)
    n_objects = (~np.isnan(xy[:, :, 0])).sum(axis=1)
    ball_seen = ~np.isnan(ball[:, 0])
    rows = (("antes do jogo", 0, first1 - 1),
            ("1º tempo (graduado)", first1, last1),
            ("intervalo", last1 + 1, first2 - 1),
            ("2º tempo, antes do apito (graduado, todo D)", first2, KICKOFF_2ND - 1),
            ("2º tempo, em jogo (graduado)", KICKOFF_2ND, last2),
            ("depois do jogo", last2 + 1, n_frames - 1))
    print("  %-44s %8s %9s %10s %8s" % ("segmento", "frames", "objs/frame", "frames 0 obj", "bola"))
    for name, first, last in rows:
        last = min(last, n_frames - 1)
        if last < first:
            continue
        sel = slice(first, last + 1)
        objs = n_objects[sel]
        print("  %-44s %8d %9.1f %9.1f%% %7.1f%%" % (
            name, len(objs), objs.mean(), 100.0 * (objs == 0).mean(),
            100.0 * ball_seen[sel].mean()))
    print("  'Quase sem objetos no intervalo' vale se objs/frame for baixo e 'frames 0 obj' alto.")
 
 
def check_values(xy, ball, graded):
    section("3. validade dos valores (x/y parciais, fora de [0,1], posição (0,0))")
    print("  A bola fora de [0,1] pode ser legítima (bola fora de campo): é pista para o D.")
    for name, data in (("jogadores", xy.reshape(-1, 2)), ("bola", ball)):
        for scope, sel in (("todos os frames", None), ("graduados", graded)):
            pos = data if sel is None else (xy[sel].reshape(-1, 2) if name == "jogadores"
                                            else ball[sel])
            nan_x, nan_y = np.isnan(pos[:, 0]), np.isnan(pos[:, 1])
            valid = pos[~nan_x & ~nan_y]
            if not len(valid):
                print("  %-9s %-16s sem valores" % (name, scope))
                continue
            outside = ((valid < 0) | (valid > 1)).any(axis=1)
            origin = (valid == 0).all(axis=1)
            print("  %-9s %-16s válidos %9d | x/y parcial %d | fora de [0,1] %d (%.2f%%) | "
                  "(0,0) exato %d | x [%.3f, %.3f] y [%.3f, %.3f]" % (
                      name, scope, len(valid), int((nan_x != nan_y).sum()), int(outside.sum()),
                      100.0 * outside.mean(), int(origin.sum()),
                      valid[:, 0].min(), valid[:, 0].max(), valid[:, 1].min(), valid[:, 1].max()))
 
 
def jumps(xym, present, team, frames, half, jump_m, max_speed):
    """Imprime a seção 4 e devolve as durações (frames) das lacunas internas dos slots."""
    section("4. saltos e reaparições (troca de identidade / fragmentação)")
    print("  salto: slot presente em 2 frames seguidos a mais de %.1f m (%.0f m/s)" % (
        jump_m, jump_m * FPS))
    print("  reaparição: slot volta após uma lacuna mais longe do que %.0f m/s + %.0f m permitem"
          % (max_speed, REAPPEAR_MARGIN_M))
    rows, all_gaps = [], []
    for slot in range(xym.shape[1]):
        idx = np.flatnonzero(present[:, slot])
        if len(idx) < 2:
            rows.append((slot, len(idx), 0, 0, 0))
            continue
        a, b = idx[:-1], idx[1:]
        same = half[a] == half[b]                 # não compara através do intervalo
        a, b = a[same], b[same]
        gap = frames[b] - frames[a]
        step = xym[b, slot] - xym[a, slot]
        disp = np.hypot(step[:, 0], step[:, 1])
        far = disp > max_speed * gap / FPS + REAPPEAR_MARGIN_M
        all_gaps.append(gap[gap > 1] - 1)         # frames ausentes dentro da lacuna
        rows.append((slot, len(idx),
                     int(np.sum((gap == 1) & (disp > jump_m))),
                     int(np.sum(gap > 1)),
                     int(np.sum((gap > 1) & far))))
    table = np.array(rows)
    for name, side in (("casa", HOME), ("fora", AWAY), ("todos", None)):
        sel = table if side is None else table[team[table[:, 0]] == side]
        frames_n, j1, reapp, reapp_far = (sel[:, 1].sum(), sel[:, 2].sum(), sel[:, 3].sum(),
                                          sel[:, 4].sum())
        print("  %-5s player-frames %8d | saltos %5d (%.1f por 100k) | reaparições %6d, "
              "impossíveis %5d (%.1f%%)" % (
                  name, frames_n, j1, 1e5 * j1 / max(frames_n, 1), reapp, reapp_far,
                  100.0 * reapp_far / max(reapp, 1)))
    print("  fragmentos por slot ativo (reaparições+1): média %.1f" % (
        (table[:, 3][table[:, 1] > 0] + 1).mean()))
    worst = table[np.argsort(-(table[:, 2] + table[:, 4]))[:5]]
    print("  slots com mais eventos suspeitos (slot, time, saltos, reaparições impossíveis):")
    for slot, _, j1, _, reapp_far in worst:
        print("    slot %2d  %-4s  %5d  %5d" % (slot + 1, "casa" if team[slot] == HOME else "fora",
                                                j1, reapp_far))
    return np.concatenate(all_gaps) if all_gaps else np.zeros(0, dtype=np.int64)
 
 
def gap_report(name, gaps):
    if not len(gaps):
        print("  %s: nenhuma lacuna" % name)
        return
    seconds = gaps / float(FPS)
    print("  %s: %d lacunas, %d frames ausentes | mediana %.2f s, p90 %.2f s, máx %.1f s" % (
        name, len(gaps), gaps.sum(), np.median(seconds), np.percentile(seconds, 90),
        seconds.max()))
    for low, high, label in GAP_BINS:
        sel = (gaps >= low) & ((gaps <= high) if high else True)
        print("      %-8s %5.1f%% das lacunas | %5.1f%% dos frames ausentes" % (
            label, 100.0 * sel.mean(), 100.0 * gaps[sel].sum() / gaps.sum()))
 
 
def gaps_section(player_gaps, ball, frames, half):
    section("5. duração das lacunas (frames graduados, dentro de cada tempo)")
    print("  jogadores: só lacunas internas (entre duas presenças do mesmo slot).")
    print("  bola: toda sequência de ausência, inclusive nas pontas de cada tempo.")
    gap_report("jogadores", player_gaps)
    absent = np.isnan(ball[frames, 0])
    ball_gaps = [run_lengths(absent[half == number]) for number in (1, 2)]
    gap_report("bola", np.concatenate(ball_gaps))
 
 
def duplicates(xym, present, team, dup_m):
    section("6. duplicatas: dois slots a menos de %.2f m no mesmo frame" % dup_m)
    n, slots = present.shape
    same_team = team[:, None] == team[None, :]
    upper = np.triu(np.ones((slots, slots), dtype=bool), 1)
    total = present.sum(axis=1)
    pair_frames = np.zeros((slots, slots), dtype=np.int64)
    frames_same = frames_big = frames_big_same = 0
    for start in range(0, n, 4000):
        sl = slice(start, min(start + 4000, n))
        chunk = xym[sl]
        diff = chunk[:, :, None, :] - chunk[:, None, :, :]
        with np.errstate(invalid="ignore"):
            close = (np.hypot(diff[..., 0], diff[..., 1]) < dup_m) & upper
        pair_frames += close.sum(axis=0)
        same_close = (close & same_team).any(axis=(1, 2))
        big = total[sl] > 22
        frames_same += int(same_close.sum())
        frames_big += int(big.sum())
        frames_big_same += int((big & same_close).sum())
    print("  pares-frame mesmo time: %d | times diferentes: %d (duelo a < %.2f m pode ser real)" % (
        int((pair_frames * same_team).sum()), int((pair_frames * ~same_team).sum()), dup_m))
    print("  frames com algum par duplicado no mesmo time: %d (%.2f%%)" % (
        frames_same, 100.0 * frames_same / n))
    print("  frames com >22 objetos: %d | com par duplicado no mesmo time: %.1f%%" % (
        frames_big, 100.0 * frames_big_same / max(frames_big, 1)))
    order = np.argsort(-pair_frames.ravel())[:5]
    print("  pares de slots mais frequentes (slot, slot, mesmo time?, frames):")
    for flat in order:
        i, j = divmod(int(flat), slots)
        if pair_frames[i, j]:
            print("    slots %2d e %2d  %-4s  %6d" % (i + 1, j + 1,
                                                    "sim" if team[i] == team[j] else "não",
                                                    pair_frames[i, j]))
 
 
def coherence(xym, present, team):
    section("7. coerência espacial: o slot está mais perto do próprio time ou do outro?")
    print("  por frame: distância do slot ao centroide do próprio time (sem ele) vs. ao do outro.")
    print("  Atacantes isolados ficam legitimamente mais perto do outro centroide em contra-ataques,")
    print("  então compare cada slot com a mediana, não com zero. Suspeito = muito acima dos pares")
    print("  ou com sequências longas.")
    zeroed = np.where(present[:, :, None], xym, 0.0)
    count = [present[:, team == side].sum(axis=1) for side in (HOME, AWAY)]
    total = [zeroed[:, team == side].sum(axis=1) for side in (HOME, AWAY)]
    rows = []
    for slot in range(xym.shape[1]):
        mine = int(team[slot])
        other = 1 - mine
        ok = present[:, slot] & (count[mine] >= 2) & (count[other] >= 1)
        if not ok.any():
            continue
        own = (total[mine] - zeroed[:, slot]) / np.maximum(count[mine] - 1, 1)[:, None]
        opp = total[other] / np.maximum(count[other], 1)[:, None]
        d_own = np.hypot(*(xym[:, slot] - own).T)
        d_opp = np.hypot(*(xym[:, slot] - opp).T)
        wrong = ok & (d_opp < d_own)
        rows.append((slot, mine, int(ok.sum()), wrong[ok].mean(), longest_run(wrong) / float(FPS)))
    usable = [r for r in rows if r[2] >= 1000]
    if not usable:
        print("  nenhum slot com >= 1000 frames")
        return
    fractions = np.array([r[3] for r in usable])
    print("  slots com >= 1000 frames: %d | fração 'mais perto do outro': mediana %.1f%%, "
          "p90 %.1f%%, máx %.1f%%" % (len(usable), 100 * np.median(fractions),
                                      100 * np.percentile(fractions, 90), 100 * fractions.max()))
    print("  top 8 por fração (slot, time, frames presentes, fração, maior sequência em s):")
    for slot, mine, n, frac, run in sorted(usable, key=lambda r: -r[3])[:8]:
        print("    slot %2d  %-4s  %7d  %5.1f%%  %6.1f s" % (
            slot + 1, "casa" if mine == HOME else "fora", n, 100 * frac, run))
 
 
def side_flip(xym, present, team, half):
    section("8. os lados espelham no 2º tempo?")
    mask = team == HOME
    n = present[:, mask].sum(axis=1)
    cx = np.where(present[:, mask], xym[:, mask, 0], 0.0).sum(axis=1) / np.maximum(n, 1)
    cx = cx / PITCH[0]
    values = {}
    for number in (1, 2):
        sel = (half == number) & (n >= 8)
        values[number] = cx[sel].mean()
        print("  centroide x médio da casa, %do tempo: %.3f" % (number, values[number]))
    mirrored = abs(values[1] - (1 - values[2])) < abs(values[1] - values[2])
    print("  -> %s" % ("valores espelhados (a <-> 1-a): os lados TROCAM no 2º tempo; features de "
                       "'atacar para a direita' precisam inverter x." if mirrored else
                       "valores parecidos: as coordenadas NÃO parecem espelhar (confirme olhando "
                       "o kickoff do 2º tempo, frame %d)." % KICKOFF_2ND))
 
 
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jump-m", type=float, default=2.0,
                        help="deslocamento (m) entre frames consecutivos tratado como salto")
    parser.add_argument("--max-speed", type=float, default=10.0,
                        help="velocidade máxima (m/s) plausível ao julgar reaparições")
    parser.add_argument("--dup-m", type=float, default=0.5,
                        help="distância (m) abaixo da qual dois slots contam como duplicata")
    parser.add_argument("--no-cache", action="store_true", help="não ler/escrever o cache")
    args = parser.parse_args(argv)
 
    xy, ball, team = load(not args.no_cache)
    half_all = half_of_frames(len(xy))
    graded = half_all > 0
    frames = np.flatnonzero(graded)                # índices de vídeo dos frames graduados
    half = half_all[frames]
 
    check_ball(ball, graded)
    xym = xy[frames] * np.array(PITCH, dtype=np.float32)     # metros, só frames graduados
    present = ~np.isnan(xym[:, :, 0])
 
    team_counts(present, team)
    segments(xy, ball)
    check_values(xy, ball, graded)
    player_gaps = jumps(xym, present, team, frames, half, args.jump_m, args.max_speed)
    gaps_section(player_gaps, ball, frames, half)
    duplicates(xym, present, team, args.dup_m)
    coherence(xym, present, team)
    side_flip(xym, present, team, half)
 
 
if __name__ == "__main__":
    main()
 