"""O portão do posse/predictor.py, testado em 2 jogos da Bundesliga com gabarito por frame da DFL.

    DFL_DIR=/caminho/para/os/xml python validacao_metrica/testar_portao_dfl.py

Variantes: baseline + D, Samuel (portão ligado) e Samuel sem portão, nos 3 cenários de
testar_predictor_samuel.py. Dois gabaritos:

    DFL bruto        BallStatus / BallPossession como vêm
    DFL sem toques   o mesmo, com as posses de menos de 1 s devolvidas ao time de antes. O CHALLENGE.md diz
                     que um toque sem controle não troca a posse; uma "posse" de < 1 s é quase sempre isso.
                     Serve para ver se a conclusão sobre o portão depende dessa definição.
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

import dfl  # noqa: E402
from testar_dono_reinicio import (combine_with_baseline, degradar, deslocar, mais_proximo,  # noqa: E402
                                  mascara_d, pontuar, smooth)
from testar_predictor_samuel import samuel  # noqa: E402

FPS = 25
JOGOS = ("J03WN1", "J03WOH")
SEMENTES = range(3)
MIN_POSSE_S = 1.0
OUT = os.path.join(HERE, "out")


def sem_toques(gt, min_s=MIN_POSSE_S):
    """Posses H/A de menos de min_s entre duas posses do mesmo time (ou depois de uma) voltam ao time de antes."""
    g = gt.copy()
    idx = np.flatnonzero(np.concatenate(([True], g[1:] != g[:-1])))
    runs = [(s, e, g[s]) for s, e in zip(idx, np.append(idx[1:], len(g)))]
    for k in range(1, len(runs)):
        s, e, lab = runs[k]
        prev = runs[k - 1][2]
        if lab in ("H", "A") and prev in ("H", "A") and lab != prev and e - s < min_s * FPS:
            g[s:e] = prev
            runs[k] = (s, e, prev)
    return g


def estatisticas(gt):
    v = gt != "X"
    lab = gt[v]
    idx = np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1])))
    lens = np.diff(np.append(idx, len(lab)))
    ha = np.isin(lab[idx], ["H", "A"])
    return dict(posses=int(ha.sum()), media_s=lens[ha].mean() / FPS, curtas_1s=int((lens[ha] < FPS).sum()))


def main():
    os.makedirs(OUT, exist_ok=True)
    linhas = []
    for m in JOGOS:
        J = dfl.carregar_jogo_dfl(m)
        bruto = J.gt.copy()
        for gnome, gt in (("DFL bruto", bruto), ("DFL sem toques", sem_toques(bruto))):
            J.gt = gt
            print(m, gnome, estatisticas(gt), flush=True)
            vivo = np.isin(gt, ["H", "A"])
            P = J.reinicios[["inicio", "fim", "tipo"]]
            cenarios = [("bola completa, D exato", [(J.ball, P)]),
                        ("bola apagada, D exato", [(degradar(J.ball, vivo, seed=s), P) for s in SEMENTES]),
                        ("bola apagada, D ±2 s", [(degradar(J.ball, vivo, seed=s), deslocar(P, s)) for s in SEMENTES])]
            for cnome, casos in cenarios:
                res = {}
                for ball, p in casos:
                    graded = gt != "X"
                    dead = mascara_d(p, J.N) & graded
                    r = {"baseline + D": pontuar(gt, combine_with_baseline(smooth(mais_proximo(J, ball)), dead, graded)),
                         "samuel": pontuar(gt, samuel(J, ball, dead, gate=True)),
                         "samuel sem portão": pontuar(gt, samuel(J, ball, dead, gate=False))}
                    for k, v in r.items():
                        res.setdefault(k, []).append(v)
                for k, v in res.items():
                    linhas.append(dict(jogo=m, gabarito=gnome, cenario=cnome, variante=k,
                                       spellF1=np.mean([x[0] for x in v]), macroF1=np.mean([x[1] for x in v])))
                print("  {:<24} ".format(cnome) + " | ".join(
                    "{} {:.3f}".format(k, np.mean([x[0] for x in v])) for k, v in res.items()), flush=True)
        J.gt = bruto
    T = pd.DataFrame(linhas)
    T.to_csv(os.path.join(OUT, "portao_dfl.csv"), index=False)
    print()
    print(T.pivot_table(index=["gabarito", "cenario", "variante"], columns="jogo", values="spellF1").round(3).to_string())


if __name__ == "__main__":
    main()
