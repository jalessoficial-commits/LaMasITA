"""Compara a posse H/A (analise/posse.py) por fonte de bola e parâmetros, contra dois gabaritos.

    python analise/comparar_posse.py                    # grava analise/out/posse/comparacao.csv
    python analise/comparar_posse.py --exportar NOME    # e a config PADRAO em evaluation/results/NOME.csv
                                                        # e viewer/data/pred_NOME.csv

Gabaritos (nenhum é usado para treinar; a grade é pequena de propósito):
    publico   GT público (5 min, evaluation/gt)
    blocos    os blocos rotulados no viewer (rotular_paradas.gt_blocos)

Em cada gabarito, dois D:
    D nosso   o detector v2 (analise/out/tracking/D_<variante>.csv) -- o que vai para a submissão
    D oráculo o D do próprio gabarito -- isola o erro de H/A do erro de D
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import fontes_bola as B  # noqa: E402
import posse as PO  # noqa: E402
import rotular_paradas as RP  # noqa: E402
import rotulos as R  # noqa: E402
from features import load_atd  # noqa: E402

OUT = os.path.join(HERE, "out", "posse")
D_VARIANTE = "atd_bruto_fora_uniao"
FONTES = ("atd_bruto", "fundida", "kalman_pred")
PADRAO = ("fundida", 50.0, 0.0)            # a escolhida: a mais simples entre as melhores
GRADE = dict(custo_troca=(25.0, 35.0, 50.0, 70.0, 100.0), margem_peso_m=(0.0, 1.0, 3.0))
# a histerese (custo_troca=None) com raio_m, k, janela foi testada antes e ficou abaixo do baseline


def carregar_D(variante=D_VARIANTE):
    return pd.read_csv(os.path.join(HERE, "out", "tracking", "D_{}.csv".format(variante)))["morta"].values.astype(bool)


def pontuar(gt, pred):
    r = T.evaluate(gt, np.where(gt == "X", "X", pred))
    return r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]


def acc_time(gt, pred):
    m = np.isin(gt, ["H", "A"]) & np.isin(pred, ["H", "A"])
    return float((gt[m] == pred[m]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exportar", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    players, _, slot_teams, _ = load_atd()
    N = len(players)
    graded = R.graded_mask(N)
    dead = carregar_D()
    gts = {"publico": R.gt_public(), "blocos": RP.gt_blocos(N)}
    print({k: dict(zip(*np.unique(v[v != "X"], return_counts=True))) for k, v in gts.items()})

    rows = []
    base = T.load_baseline("baseline_smooth125")
    for g, gt in gts.items():
        for dnome, d in (("nosso", dead), ("oraculo", gt == "D")):
            p = T.combine_with_baseline(base, d, graded)
            s, f, ng, npred = pontuar(gt, p)
            rows.append(dict(fonte="baseline_smooth125", gabarito=g, D=dnome, spellF1=s, macroF1=f,
                             acc_HA=acc_time(gt, p), n_gt=ng, n_pred=npred))

    for fonte in FONTES:
        ball = B.carregar_bola(fonte)
        qual = B.extras_fundida()[0] if fonte == "fundida" else None
        dists = PO.distancias(ball, players, slot_teams)
        for vals in itertools.product(*GRADE.values()):
            cfg = PO.Config(**dict(zip(GRADE, vals)))
            for g, gt in gts.items():
                for dnome, d in (("nosso", dead), ("oraculo", gt == "D")):
                    p = PO.prever(ball, players, slot_teams, d, graded, cfg, qual, dists)
                    s, f, ng, npred = pontuar(gt, p)
                    rows.append(dict(fonte=fonte, gabarito=g, D=dnome, spellF1=s, macroF1=f,
                                     acc_HA=acc_time(gt, p), n_gt=ng, n_pred=npred, **dict(zip(GRADE, vals))))
        print(fonte, "ok", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "comparacao.csv"), index=False)
    chave = ["fonte"] + list(GRADE)
    wide = res.pivot_table(index=chave, columns=["gabarito", "D"], values="spellF1", dropna=False)
    wide.columns = ["{}_{}".format(a, b) for a, b in wide.columns]
    wide["media_nosso"] = wide[["publico_nosso", "blocos_nosso"]].mean(1)
    wide = wide.sort_values("media_nosso", ascending=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(wide.round(3).head(25).to_string())
    print(res[res.fonte == "baseline_smooth125"].round(3).to_string(index=False))

    if args.exportar:
        fonte, vals = PADRAO[0], PADRAO[1:]
        cfg = PO.Config(**dict(zip(GRADE, vals)))
        ball = B.carregar_bola(fonte)
        qual = B.extras_fundida()[0] if fonte == "fundida" else None
        lab = PO.prever(ball, players, slot_teams, dead, graded, cfg, qual)
        path = os.path.join(T.ROOT, "evaluation", "results", args.exportar + ".csv")
        pd.DataFrame({"frame": np.arange(N), "label": lab}).to_csv(path, index=False)
        vpath = os.path.join(T.ROOT, "viewer", "data", "pred_{}.csv".format(args.exportar))
        pd.DataFrame({"frame": np.arange(N), "label": lab}).to_csv(vpath, index=False)  # para o viewer
        print("padrão:", fonte, cfg, "->", path)


if __name__ == "__main__":
    main()
