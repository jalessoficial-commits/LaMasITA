"""Escolhe a bola usando a continuidade no tempo: filtro de Kalman ou Viterbi sobre os candidatos do jogo todo.

Colocar em tracking_bola/, junto de run_champion_full.py, compare_trackers.py, eval_atd.py, check_projection.py
e mask_pitch.py. Só numpy. Usa a saída do run_champion_full.py (out/full_rf_campo.csv: candidatos do Roboflow
já com a máscara do campo, em todos os frames avaliados) e o ATD:

    python tracking_bola/track_temporal.py --method kalman      # -> out/kalman_bola.csv
    python tracking_bola/track_temporal.py --method viterbi     # -> out/viterbi_bola.csv

    python tracking_bola/compare_trackers.py --tracker campeao=tracking_bola/out/full_bola.csv --tracker kalman=tracking_bola/out/kalman_bola.csv --tracker viterbi=tracking_bola/out/viterbi_bola.csv --tol 40
    (e o mesmo com --annotations .../ball_annotations_teste.csv para o conjunto de teste)

A fusão campeã decide cada frame sozinha. Aqui a decisão usa os frames vizinhos. Em cada frame os
candidatos são: os do detector (conf >= --floor) e a bola do ATD. Tudo é feito em METROS no campo (inversa
da homografia), onde "a bola não pula 10 m em 1/25 s" faz sentido.

KALMAN (--method kalman): rastro único, só olha o passado. Estado = posição e velocidade; a cada frame
prevê a posição (velocidade constante, com ruído de aceleração --accel) e aceita só candidatos dentro da
janela estatística da previsão (--gate, distância de Mahalanobis ao quadrado; 9,21 = 99% em 2D). Entre os
aceitos, fica o de menor custo = distância² + --conf-weight x (1 - conf). Sem candidato aceito, segue a
previsão por até --max-coast frames ("previsto"); depois desiste. Se, fora da janela, aparecer um candidato
confiante (conf >= --init-conf) em 2 frames seguidos e coerentes entre si, o rastro recomeça nele (a bola
mudou de rumo ou o rastro estava errado). O rastro começa (ou recomeça) num
candidato do detector com conf >= --init-conf ou, se não houver, na bola do ATD.

VITERBI (--method viterbi): olha o trecho inteiro (passado e futuro). Em cada frame, os estados possíveis
são os candidatos e "sem bola". Escolhe a sequência de estados de menor custo total:
  custo de um candidato do detector = --conf-weight x (1 - conf); da bola do ATD = --atd-cost;
  "sem bola" = --none-cost por frame;
  passar de uma bola a outra no frame seguinte = --move-cost x distância (m), proibido acima de --vmax m/s;
  entrar ou sair de "sem bola" = --switch-cost.
Calibragem dos padrões: uma detecção isolada (1 frame, entre frames sem bola) só vence "sem bola" se
1 - conf + 2 x 0,15 < 0,5, isto é, conf > 0,8 (o mesmo limiar da fusão campeã); uma sequência de n frames
seguidos e coerentes vence com conf > 0,5 + 0,3/n (por exemplo, 0,56 com n = 5). A bola do ATD (custo 0,3)
sempre vence "sem bola", como na fusão. É assim que a continuidade aceita detecções de confiança média.
Depois, lacunas de até --max-fill frames entre duas bolas são preenchidas em linha reta ("interpolado"),
se o salto couber em --vmax.

Nos dois: se a posição escolhida é a do ATD e há um candidato do detector a até --radius px dela, vale a do
detector ("detector_refinou"), como na fusão campeã.

Saída: frame,x,y,regra em pixels do vídeo (só frames com bola), no formato que o compare_trackers.py lê.
regra = detector, atd, detector_refinou, previsto (Kalman) ou interpolado (Viterbi).
Os parâmetros padrão são um ponto de partida; ajuste nos 292 frames de validação e decida pelos de TESTE.
DADO DERIVADO DO VÍDEO: fica em out/, fora do Git.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from check_projection import read_ball  # noqa: E402
from eval_atd import atd_pixel  # noqa: E402
from compare_trackers import load_tracker  # noqa: E402
from mask_pitch import pitch_size  # noqa: E402

DT = 1.0 / 25


class Candidate(object):
    __slots__ = ("px", "m", "conf", "kind")

    def __init__(self, px, m, conf, kind):
        self.px, self.m, self.conf, self.kind = px, np.asarray(m, dtype=float), conf, kind


def build_candidates(frames, every, ball, matrices, valid, length, width, floor):
    """{frame: [Candidate]} com os candidatos do detector (conf >= floor) e a bola do ATD, em pixels e metros."""
    result = {}
    for frame in frames:
        found = []
        if valid[frame]:
            inverse = np.linalg.inv(matrices[frame])
            for x, y, conf in every.get(frame, []):
                if conf >= floor:
                    u, v = project(inverse, x, y)
                    found.append(Candidate((x, y), (float(u) * length, float(v) * width), conf, "detector"))
            atd = atd_pixel(ball, matrices, valid, frame)
            if atd is not None:
                found.append(Candidate(atd, (ball[frame, 0] * length, ball[frame, 1] * width), None, "atd"))
        result[frame] = found
    return result


def refine(frame, chosen, candidates, radius):
    """Se escolheu o ATD e há candidato do detector a até radius px, troca pela posição do detector."""
    if chosen.kind != "atd":
        return chosen.px, "detector"
    near = [c for c in candidates[frame] if c.kind == "detector"
            and np.hypot(c.px[0] - chosen.px[0], c.px[1] - chosen.px[1]) <= radius]
    if near:
        best = max(near, key=lambda c: c.conf)
        return best.px, "detector_refinou"
    return chosen.px, "atd"


def run_kalman(frames, candidates, to_pixel, args):
    F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    q = args.accel ** 2
    Q = q * np.array([[DT ** 4 / 4, 0, DT ** 3 / 2, 0], [0, DT ** 4 / 4, 0, DT ** 3 / 2],
                      [DT ** 3 / 2, 0, DT ** 2, 0], [0, DT ** 3 / 2, 0, DT ** 2]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    out, state, cov, misses, pending = {}, None, None, 0, None
    jump = args.vmax * DT

    def noise(c):
        sigma = args.det_sigma if c.kind == "detector" else args.atd_sigma
        return np.eye(2) * sigma ** 2

    def confidence(c):
        return c.conf if c.kind == "detector" else args.atd_conf

    for frame in frames:
        options = candidates.get(frame, [])
        if state is not None:
            state = F @ state
            cov = F @ cov @ F.T + Q
            best, best_cost, best_S = None, None, None
            for c in options:
                S = H @ cov @ H.T + noise(c)
                residual = c.m - H @ state
                d2 = float(residual @ np.linalg.solve(S, residual))
                if d2 <= args.gate:
                    cost = d2 + args.conf_weight * (1.0 - confidence(c))
                    if best is None or cost < best_cost:
                        best, best_cost, best_S = c, cost, S
            if best is not None:
                gain = cov @ H.T @ np.linalg.inv(best_S)
                state = state + gain @ (best.m - H @ state)
                cov = (np.eye(4) - gain @ H) @ cov
                misses, pending = 0, None
                out[frame] = refine(frame, best, candidates, args.radius)
                continue
            # bola confiante fora da janela em 2 frames seguidos e coerentes entre si: o rastro estava
            # errado (ou a bola mudou de rumo); recomeça nela em vez de seguir a previsão
            strong = [c for c in options if c.kind == "detector" and c.conf >= args.init_conf]
            restart = None
            if pending is not None and strong:
                close = [c for c in strong if np.hypot(*(c.m - pending.m)) <= jump]
                restart = max(close, key=lambda c: c.conf) if close else None
            pending = max(strong, key=lambda c: c.conf) if strong else None
            if restart is not None:
                state, misses, pending = None, 0, None
            else:
                misses += 1
            if state is not None and misses <= args.max_coast:
                pixel = to_pixel(frame, state[0], state[1])
                if pixel is not None:
                    out[frame] = (pixel, "previsto")
                continue
            state = None
        pending = None
        strong = [c for c in options if c.kind == "detector" and c.conf >= args.init_conf]
        start = max(strong, key=lambda c: c.conf) if strong else next((c for c in options if c.kind == "atd"), None)
        if start is not None:
            state = np.array([start.m[0], start.m[1], 0.0, 0.0])
            sigma = args.det_sigma if start.kind == "detector" else args.atd_sigma
            cov = np.diag([sigma ** 2, sigma ** 2, args.init_speed ** 2, args.init_speed ** 2])
            misses = 0
            out[frame] = refine(frame, start, candidates, args.radius)
    return out


def run_viterbi(frames, candidates, args):
    """frames: um trecho contínuo. Devolve {frame: ((x, y), regra)}."""
    jump = args.vmax * DT
    costs, backs, states = [], [], []
    previous = None
    for index, frame in enumerate(frames):
        options = candidates.get(frame, [])
        emission = np.array([args.conf_weight * (1.0 - c.conf) if c.kind == "detector" else args.atd_cost
                             for c in options] + [args.none_cost])
        if previous is None:
            total = emission.copy()
            back = np.full(len(emission), -1)
        else:
            prev_cost, prev_options = previous
            k_prev, k_now = len(prev_options), len(options)
            trans = np.full((k_prev + 1, k_now + 1), np.inf)
            if k_prev and k_now:
                a = np.array([c.m for c in prev_options])
                b = np.array([c.m for c in options])
                distance = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
                trans[:k_prev, :k_now] = np.where(distance <= jump, args.move_cost * distance, np.inf)
            trans[:k_prev, k_now] = args.switch_cost      # bola -> sem bola
            trans[k_prev, :k_now] = args.switch_cost      # sem bola -> bola
            trans[k_prev, k_now] = 0.0                    # sem bola -> sem bola
            candidates_cost = prev_cost[:, None] + trans
            back = np.argmin(candidates_cost, axis=0)
            total = candidates_cost[back, np.arange(len(emission))] + emission
        costs.append(total)
        backs.append(back)
        states.append(options)
        previous = (total, options)

    path = [int(np.argmin(costs[-1]))]
    for index in range(len(frames) - 1, 0, -1):
        path.append(int(backs[index][path[-1]]))
    path.reverse()

    out = {}
    chosen = [states[i][k] if k < len(states[i]) else None for i, k in enumerate(path)]
    for i, frame in enumerate(frames):
        if chosen[i] is not None:
            out[frame] = refine(frame, chosen[i], candidates, args.radius)
    # lacunas curtas entre duas bolas: linha reta em pixels, se o salto for fisicamente possível
    i = 0
    while i < len(frames):
        if chosen[i] is not None:
            i += 1
            continue
        end = i
        while end < len(frames) and chosen[end] is None:
            end += 1
        if i > 0 and end < len(frames) and end - i <= args.max_fill:
            a, b = chosen[i - 1], chosen[end]
            steps = end - i + 1
            if np.hypot(*(a.m - b.m)) <= jump * steps:
                pa, pb = out[frames[i - 1]][0], out[frames[end]][0]
                for j in range(i, end):
                    t = (j - i + 1) / float(steps)
                    out[frames[j]] = ((pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])), "interpolado")
        i = end
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["kalman", "viterbi"], required=True)
    parser.add_argument("--candidates", default=os.path.join(HERE, "out", "full_rf_campo.csv"))
    parser.add_argument("--out", default=None, help="padrão: out/<method>_bola.csv")
    parser.add_argument("--floor", type=float, default=0.2, help="conf mínima dos candidatos (padrão 0.2)")
    parser.add_argument("--radius", type=float, default=40.0, help="refino ATD -> detector, px (padrão 40)")
    parser.add_argument("--conf-weight", type=float, default=1.0, help="peso de (1 - conf) no custo (padrão 1)")
    # Kalman
    parser.add_argument("--accel", type=float, default=20.0, help="Kalman: ruído de aceleração, m/s² (padrão 20)")
    parser.add_argument("--det-sigma", type=float, default=0.3, help="Kalman: erro do detector, m (padrão 0.3)")
    parser.add_argument("--atd-sigma", type=float, default=1.0, help="Kalman: erro do ATD, m (padrão 1.0)")
    parser.add_argument("--atd-conf", type=float, default=0.6, help="Kalman: 'conf' atribuída ao ATD (padrão 0.6)")
    parser.add_argument("--gate", type=float, default=9.21, help="Kalman: janela, Mahalanobis² (padrão 9.21 = 99%%)")
    parser.add_argument("--max-coast", type=int, default=12, help="Kalman: frames seguindo só a previsão (padrão 12)")
    parser.add_argument("--init-conf", type=float, default=0.8, help="Kalman: conf para começar um rastro (padrão 0.8)")
    parser.add_argument("--init-speed", type=float, default=10.0, help="Kalman: incerteza inicial da velocidade, m/s")
    # Viterbi
    parser.add_argument("--atd-cost", type=float, default=0.3, help="Viterbi: custo da bola do ATD (padrão 0.3)")
    parser.add_argument("--none-cost", type=float, default=0.5, help="Viterbi: custo de 'sem bola' por frame (padrão 0.5)")
    parser.add_argument("--move-cost", type=float, default=0.1, help="Viterbi: custo por metro entre frames (padrão 0.1)")
    parser.add_argument("--switch-cost", type=float, default=0.15, help="Viterbi: custo de entrar/sair de 'sem bola' (padrão 0.15)")
    parser.add_argument("--vmax", type=float, default=45.0, help="velocidade máxima plausível, m/s (padrão 45)")
    parser.add_argument("--max-fill", type=int, default=12, help="Viterbi: maior lacuna preenchida, frames (padrão 12)")
    args = parser.parse_args(argv)
    out_path = args.out or os.path.join(HERE, "out", "{}_bola.csv".format(args.method))
    if not os.path.isfile(args.candidates):
        sys.exit("não achei {}. Rode o run_champion_full.py antes.".format(args.candidates))

    n_frames = source_frames("tactical", GAME)
    matrices, _, valid = load_homography()
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    length, width = pitch_size()
    _, _, every = load_tracker(args.candidates)
    blocks = [list(range(first, last + 1)) for _, first, last in half_video_blocks(GAME)]
    all_frames = [f for block in blocks for f in block]
    candidates = build_candidates(all_frames, every, ball, matrices, valid, length, width, args.floor)

    def to_pixel(frame, xm, ym):
        if not valid[frame]:
            return None
        x, y = project(matrices[frame], xm / length, ym / width)
        return float(x), float(y)

    result = {}
    for block in blocks:
        if args.method == "kalman":
            result.update(run_kalman(block, candidates, to_pixel, args))
        else:
            result.update(run_viterbi(block, candidates, args))

    folder = os.path.dirname(out_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    counts = {}
    with open(out_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "regra"])
        for frame in sorted(result):
            (x, y), rule = result[frame]
            counts[rule] = counts.get(rule, 0) + 1
            writer.writerow([frame, "{:.2f}".format(x), "{:.2f}".format(y), rule])
    total = len(all_frames)
    print("{}: bola em {} de {} frames avaliados ({:.1f}%)".format(args.method, len(result), total, 100.0 * len(result) / total))
    for rule in sorted(counts):
        print("  {:<18} {:7d}".format(rule, counts[rule]))
    print("saída: {}".format(out_path))


if __name__ == "__main__":
    main()