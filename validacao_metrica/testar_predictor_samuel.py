"""O posse/predictor.py (Samuel) e o analise/posse.py (Nuno) no banco de teste dos jogos da Metrica.

    METRICA_DIR=/caminho/para/sample-data python validacao_metrica/testar_predictor_samuel.py

Reproduz o main() do predictor.py com os dados da Metrica no lugar do ATD, sem ler arquivo nenhum do
jogo do desafio. Diferenças inevitáveis:
    - sem Smart Tagging (estes jogos não têm): o restart_owner decide só pelo tracking
    - sem a correção dos goleiros: ela usa os kickoffs do BAR-ATH, e na Metrica cada coluna é um
      jogador fixo, então o problema que ela corrige (slot de goleiro trocando de pessoa) não existe
    - sem a bola no ar (--aerial), que depende da homografia do vídeo do desafio
Os parâmetros são os do repositório, sem ajuste nenhum a estes jogos.

Variantes (todas com o mesmo D, do gabarito ou deslocado em até ±2 s):
    baseline + D            o baseline oficial com o D por cima
    samuel                  predictor.py padrão: portão ligado, toque 0,5
    samuel sem portão       --no-gate: toque 1,5
    samuel + dono_reinicio  o padrão, com o dono de analise/dono_reinicio.py no lugar do restart_owner
    nuno c25 / c50          analise/posse.py (branch-NUNO) com custo de troca 25 e 50 (o escolhido)
"""

import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "analise"), os.path.join(ROOT, "posse")):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrica as M  # noqa: E402
import dono_reinicio as DR  # noqa: E402
import predictor as SP  # noqa: E402
import restart_owner as ro  # noqa: E402
from testar_dono_reinicio import (combine_with_baseline, degradar, deslocar, mais_proximo,  # noqa: E402
                                  mascara_d, pontuar, smooth)

FPS = 25
PITCH = np.array([M.LENGTH, M.WIDTH])
SEMENTES = range(3)
OUT = os.path.join(HERE, "out")


def carregar_nuno():
    """analise/posse.py da branch-NUNO, lido do git (ainda não está na main)."""
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "origin/branch-NUNO:analise/posse.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    spec = importlib.util.spec_from_loader("posse_nuno", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "posse_nuno.py", "exec"), mod.__dict__)
    return mod


def blocos(gt):
    """[(metade, primeiro, último)] dos trechos avaliados (gt != X), como half_video_blocks."""
    g = (gt != "X").astype(np.int8)
    e = np.diff(np.concatenate(([0], g, [0])))
    return [(i + 1, int(s), int(t) - 1) for i, (s, t) in
            enumerate(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))]


def samuel(J, ball_m_raw, dead, gate=True, donos=None):
    """O main() do predictor.py, com os arrays da Metrica. donos: DataFrame de dono_reinicio para
    trocar o restart_owner (casado pelo frame do reinício)."""
    blocks = blocos(J.gt)
    segments = ro.live_segments(dead, blocks)
    team = np.where(J.time_slot == "H", ro.HOME, ro.AWAY).astype(np.int8)
    players = J.players / PITCH                                     # normalizado, como o ATD
    clean_m = SP.despike(ball_m_raw.astype(float), SP.SPIKE_SPAN, SP.SPIKE_DIST)
    held = ro.hold_ball(clean_m / PITCH, ro.LOOKBACK)
    view = SP.PlayersView(players)
    rows = [ro.decide(s, e, view, team, held, tuple(PITCH), []) for _, s, e in segments]
    if donos is not None:
        por_fim = {int(r.fim): r.dono for r in donos.itertuples()}
        for (_, s, _), row in zip(segments, rows):
            d = por_fim.get(s)
            if d in ("H", "A"):
                row["owner"], row["source"] = d, "tracking"

    ball_m = SP.smooth(clean_m, SP.SMOOTH)
    players_m = J.players.astype(np.float32)
    d_home, d_away = SP.team_distances(players_m, ball_m, team)
    dv, v_in, v_out = SP.velocity_change(ball_m, segments, FPS)
    touches, threshold, noise, _ = SP.detect_touches(dv, v_in, v_out, d_home, d_away, segments,
                                                     flight_filter=gate)
    allow = None
    if gate:
        allow, _ = SP.gate_mask(dv, v_in, v_out, d_home, d_away, ball_m, segments, noise)
    ev = SP.proximity_evidence(d_home, d_away, None)
    w = SP.TOUCH_WEIGHT_GATED if gate else SP.TOUCH_WEIGHT
    for t, side, _ in touches:
        ev[side, t] += w
    live = SP.predict(segments, ev, SP.last_touch(touches, segments, J.N), rows, allow)
    labels = np.full(J.N, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = "D"
    for (s, e), piece in live.items():
        labels[s:e] = piece
    return labels


def rodar(J, ball, paradas, PO):
    graded = J.gt != "X"
    dead = mascara_d(paradas, J.N) & graded
    raw = mais_proximo(J, ball)
    base = smooth(raw)
    res = {"baseline + D": pontuar(J.gt, combine_with_baseline(base, dead, graded))}
    res["samuel"] = pontuar(J.gt, samuel(J, ball, dead, gate=True))
    res["samuel sem portão"] = pontuar(J.gt, samuel(J, ball, dead, gate=False))
    donos = DR.estimar_donos(paradas[["inicio", "fim"]], J.players, J.time_slot, ball, rotulos=base)
    res["samuel + dono_reinicio"] = pontuar(J.gt, samuel(J, ball, dead, gate=True, donos=donos))
    if PO is not None:
        dists = PO.distancias(ball, J.players, J.time_slot)
        for c in (25.0, 50.0):
            res["nuno c{:.0f}".format(c)] = pontuar(J.gt, PO.prever(
                ball, J.players, J.time_slot, dead, graded, PO.Config(custo_troca=c), dists=dists))
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    PO = carregar_nuno()
    linhas = []
    for g in (1, 2):
        J = M.carregar_jogo(g)
        vivo = np.isin(J.gt, ["H", "A"])
        P = J.reinicios[["inicio", "fim", "tipo"]]
        cenarios = [("bola Metrica, D exato", [(J.ball, P)]),
                    ("bola apagada, D exato", [(degradar(J.ball, vivo, seed=s), P) for s in SEMENTES]),
                    ("bola apagada, D ±2 s", [(degradar(J.ball, vivo, seed=s), deslocar(P, s)) for s in SEMENTES])]
        for nome, casos in cenarios:
            out = [rodar(J, b, p, PO) for b, p in casos]
            for v in out[0]:
                linhas.append(dict(jogo=g, cenario=nome, variante=v,
                                   spellF1=np.mean([o[v][0] for o in out]),
                                   macroF1=np.mean([o[v][1] for o in out])))
            print("jogo {} | {:<22} ".format(g, nome) + " | ".join(
                "{} {:.3f}".format(v, np.mean([o[v][0] for o in out])) for v in out[0]), flush=True)
    T = pd.DataFrame(linhas)
    T.to_csv(os.path.join(OUT, "predictor_samuel.csv"), index=False)
    print()
    print(T.pivot_table(index=["cenario", "variante"], columns="jogo",
                        values=["spellF1", "macroF1"]).round(3).to_string())


if __name__ == "__main__":
    main()
