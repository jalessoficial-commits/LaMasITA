"""O posse/restart_owner.py (Samuel) no mesmo teste do analise/dono_reinicio.py, nos jogos da Metrica.

    METRICA_DIR=/caminho/para/sample-data python validacao_metrica/testar_restart_owner.py

Mesmos jogos, cenários e scorer de testar_dono_reinicio.py. Para o dono de cada parada compara:

    samuel             restart_owner.decide sem Smart Tagging (estes jogos não têm), com a bola segurada
                       por até 5 s como no script dele (hold_ball)
    dono_reinicio      analise/dono_reinicio.estimar_donos
    samuel + chute     o do Samuel e, quando ele não decide, o último chute antes da saída (bola fora)
                       ou a posse antes do apito, de dono_reinicio

O spellF1 usa o mesmo mecanismo para todos (o dono nos 2 s depois do reinício, antes da suavização do
baseline), então a diferença entre as linhas é só a qualidade do dono. No predictor.py o dono é o estado
inicial do Viterbi de cada trecho vivo; ali também o que conta é acertar o dono.
"""

import os
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
import restart_owner as ro  # noqa: E402
from testar_dono_reinicio import (SEMENTES, combine_with_baseline, degradar, deslocar,  # noqa: E402
                                  mais_proximo, mascara_d, pontuar, smooth)

FPS = 25
PITCH = np.array([M.LENGTH, M.WIDTH])
OUT = os.path.join(HERE, "out")


def donos_samuel(J, ball_m, paradas):
    """restart_owner.decide para cada parada: o trecho vivo vai do reinício ao início da próxima."""
    ball = ro.hold_ball(ball_m / PITCH, ro.LOOKBACK)          # ele trabalha em 0-1 e escala por pitch
    team = np.where(J.time_slot == "H", ro.HOME, ro.AWAY).astype(np.int8)
    inicios = paradas.inicio.values
    rows = []
    for i, r in enumerate(paradas.itertuples()):
        start = int(r.fim)
        end = int(inicios[i + 1]) if i + 1 < len(paradas) else J.N
        end = max(end, start + 1)
        frames = range(max(0, start - ro.PRE), min(J.N, start + ro.POST_FALLBACK))
        players = {f: J.players[f] / PITCH for f in frames}
        d = ro.decide(start, end, players, team, ball, tuple(PITCH), [])
        rows.append(dict(inicio=int(r.inicio), fim=start, dono=d["owner"] if d["owner"] in ("H", "A") else None,
                         fonte=d["source"]))
    return pd.DataFrame(rows)


def rodar(J, ball, paradas):
    graded = J.gt != "X"
    dead = mascara_d(paradas, J.N) & graded
    raw = mais_proximo(J, ball)
    base = smooth(raw)
    verdade = J.reinicios.dono.values

    nosso = DR.estimar_donos(paradas[["inicio", "fim"]], J.players, J.time_slot, ball, rotulos=base)
    sam = donos_samuel(J, ball, paradas)
    comb = sam.copy()
    falta = comb.dono.isna().values
    comb.loc[falta, "dono"] = nosso.dono.values[falta]

    res = {"baseline + D": pontuar(J.gt, combine_with_baseline(base, dead, graded))}
    acc = {}
    for nome, donos in (("samuel", sam), ("dono_reinicio", nosso), ("samuel + chute", comb)):
        res[nome] = pontuar(J.gt, combine_with_baseline(smooth(DR.aplicar_donos(raw, donos)), dead, graded))
        tem = donos.dono.isin(["H", "A"]).values
        acc[nome] = (np.mean(donos.dono.values == verdade), tem.mean(),
                     np.mean(donos.dono.values[tem] == verdade[tem]) if tem.any() else np.nan)
    certo = paradas[["inicio", "fim"]].assign(dono=verdade)
    res["dono verdadeiro"] = pontuar(J.gt, combine_with_baseline(smooth(DR.aplicar_donos(raw, certo)), dead, graded))
    fontes = sam.assign(certo=sam.dono.values == verdade, nosso_certo=nosso.dono.values == verdade)
    return res, acc, fontes


def main():
    os.makedirs(OUT, exist_ok=True)
    linhas, fontes_all = [], []
    for g in (1, 2):
        J = M.carregar_jogo(g)
        vivo = np.isin(J.gt, ["H", "A"])
        P = J.reinicios[["inicio", "fim", "tipo"]]
        cenarios = [("bola Metrica, D exato", [(J.ball, P)]),
                    ("bola apagada, D exato", [(degradar(J.ball, vivo, seed=s), P) for s in SEMENTES]),
                    ("bola apagada, D ±2 s", [(degradar(J.ball, vivo, seed=s), deslocar(P, s)) for s in SEMENTES])]
        for nome, casos in cenarios:
            out = [rodar(J, b, p) for b, p in casos]
            for v in out[0][0]:
                row = dict(jogo=g, cenario=nome, variante=v, spellF1=np.mean([o[0][v][0] for o in out]))
                if v in out[0][1]:
                    row["acerto_total"] = np.mean([o[1][v][0] for o in out])
                    row["responde"] = np.mean([o[1][v][1] for o in out])
                    row["acerto_quando_responde"] = np.mean([o[1][v][2] for o in out])
                linhas.append(row)
            fontes_all.append(pd.concat([o[2] for o in out]).assign(jogo=g, cenario=nome))
            print("jogo {} | {:<22} ".format(g, nome) + " | ".join(
                "{} {:.3f}".format(v, np.mean([o[0][v][0] for o in out])) for v in out[0][0]))
    T = pd.DataFrame(linhas)
    T.to_csv(os.path.join(OUT, "restart_owner_samuel.csv"), index=False)
    F = pd.concat(fontes_all)
    print()
    print(T.pivot_table(index=["cenario", "variante"], columns="jogo",
                        values=["spellF1", "acerto_total", "responde"]).round(3).to_string())
    print()
    print("acerto por fonte de decisão do Samuel (média dos cenários e sementes):")
    print(F.groupby(["cenario", "fonte"]).agg(n=("certo", "size"), samuel=("certo", "mean"),
                                              dono_reinicio=("nosso_certo", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
