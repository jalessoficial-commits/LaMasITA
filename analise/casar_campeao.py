"""Casa a bola morta do detector de paradas (D_v4) com o H/A do campeão do posse_ml (Jales).

    python analise/casar_campeao.py                       # tabela: campeões, v4 e os casamentos
    python analise/casar_campeao.py --exportar            # grava evaluation/results/casado_<campeão>_<D>.csv
                                                          # e viewer/data/pred_casado_<campeão>_<D>.csv

Não roda as redes de novo (a bola do Viterbi, que elas esperam, sai do detector de vídeo na GPU):
casa as duas saídas prontas, frame a frame, como o modo "rígido" do posse_ml faria.

    D do detector                 -> D
    viva no detector, H/A no campeão -> o H/A do campeão
    viva no detector, D no campeão   -> o H/A do campeão mais perto no tempo (antes ou depois)
    fora dos tempos (X)           -> o rótulo do campeão

Com o D v5 vem out/tracking/goleiro_v5.csv: os D tirados porque o goleiro segurava a bola (o ATD perde a
bola na mão dele). Ali o rótulo é o time do goleiro, não o vizinho no tempo.

Depois, uma posse curta (< --min-posse-s) espremida entre duas do outro time, ou entre um D e uma
posse do outro time, é engolida pelo vizinho: o campeão parte posses (32 contra 24 nos blocos).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import rotular_paradas as RP  # noqa: E402
import rotulos as R  # noqa: E402

FPS = R.FPS
RES = os.path.join(ROOT, "evaluation", "results")
D_DIR = os.path.join(HERE, "out", "tracking")


def ler_rotulos(caminho):
    return pd.read_csv(caminho).iloc[:, -1].values.astype(str)


def ler_D(nome):
    """D_<nome>.csv (frame,morta) de out/tracking, ou analise/out/paradas_D.csv para 'v2'."""
    caminho = os.path.join(HERE, "out", "paradas_D.csv") if nome == "v2" else os.path.join(D_DIR, "D_{}.csv".format(nome))
    return pd.read_csv(caminho).morta.values.astype(bool)


def ler_goleiro(nome):
    """Trechos (inicio, fim, time) em que o goleiro segurava a bola, se o D tiver (v5)."""
    caminho = os.path.join(D_DIR, "goleiro_{}.csv".format(nome))
    return pd.read_csv(caminho) if os.path.exists(caminho) else None


def _runs(mask):
    r = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(r == 1), np.flatnonzero(r == -1)))


def casar(campeao, morta, graded, min_posse_s=0.0, goleiro=None):
    out = campeao.copy()
    ha = np.isin(campeao, ["H", "A"])
    idx = np.flatnonzero(ha)
    # H/A do campeão mais perto no tempo, para cada frame
    pos = np.searchsorted(idx, np.arange(len(campeao)))
    antes = idx[np.clip(pos - 1, 0, len(idx) - 1)]
    depois = idx[np.clip(pos, 0, len(idx) - 1)]
    perto = np.where(np.abs(np.arange(len(campeao)) - antes) <= np.abs(depois - np.arange(len(campeao))), antes, depois)
    viva = graded & ~morta
    out[viva & ~ha] = campeao[perto[viva & ~ha]]
    out[graded & morta] = "D"
    if goleiro is not None:
        for r in goleiro.itertuples():
            m = graded[r.inicio:r.fim] & ~morta[r.inicio:r.fim]
            out[r.inicio:r.fim][m] = r.time
    if min_posse_s > 0:
        out = engolir_curtas(out, graded, int(min_posse_s * FPS))
    return out


def engolir_curtas(lab, graded, n_min):
    """Posse H/A curta que não sobrevive entre vizinhos: vira o time do vizinho que não é D."""
    lab = lab.copy()
    mudou = True
    while mudou:
        mudou = False
        trechos = []
        ini = 0
        for f in range(1, len(lab) + 1):
            if f == len(lab) or lab[f] != lab[ini]:
                trechos.append((ini, f, lab[ini]))
                ini = f
        for k in range(1, len(trechos) - 1):
            s, e, l = trechos[k]
            if l not in ("H", "A") or e - s >= n_min or not graded[s]:
                continue
            viz = [trechos[k - 1][2], trechos[k + 1][2]]
            outro = [v for v in viz if v in ("H", "A") and v != l]
            if outro and all(v in ("H", "A", "D") for v in viz):
                lab[s:e] = outro[0]
                mudou = True
    return lab


def placar(lab):
    res = {}
    for g, gt in (("publico", R.gt_public()), ("blocos", RP.gt_blocos(len(lab)))):
        r = T.evaluate(gt, lab)
        res[g + "_spellF1"] = r["spell_f1_mean"]
        res[g + "_macroF1"] = r["macro_f1"]
        res[g + "_posses"] = "{}/{}".format(r["n_pred_spells"], r["n_gt_spells"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campeoes", nargs="+", default=["campeao_tcn", "campeao_lstm"])
    ap.add_argument("--d", nargs="+", default=["v2", "v3", "v4", "v5"])
    ap.add_argument("--min-posse-s", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--exportar", action="store_true", help="grava o casamento escolhido (--d e --min-posse-s em [0])")
    args = ap.parse_args()
    graded = R.graded_mask(len(ler_rotulos(os.path.join(RES, args.campeoes[0] + ".csv"))))
    rows = []
    for c in args.campeoes:
        camp = ler_rotulos(os.path.join(RES, c + ".csv"))
        rows.append(dict(versao=c, D="próprio", min_posse_s=np.nan, **placar(camp)))
        for dn in args.d:
            morta = ler_D(dn)
            for mp in args.min_posse_s:
                lab = casar(camp, morta, graded, mp, ler_goleiro(dn))
                rows.append(dict(versao=c, D=dn, min_posse_s=mp, **placar(lab)))
    for dn in args.d:
        p = os.path.join(RES, "posse_viterbi_{}.csv".format(dn))
        if os.path.exists(p):
            rows.append(dict(versao="posse_viterbi (nosso)", D=dn, min_posse_s=np.nan, **placar(ler_rotulos(p))))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if args.exportar:
        dn, mp = args.d[0], args.min_posse_s[0]
        for c in args.campeoes:
            lab = casar(ler_rotulos(os.path.join(RES, c + ".csv")), ler_D(dn), graded, mp, ler_goleiro(dn))
            nome = "casado_{}_{}".format(c.replace("campeao_", ""), dn)
            df = pd.DataFrame({"frame": np.arange(len(lab)), "label": lab})
            df.to_csv(os.path.join(RES, nome + ".csv"), index=False)
            df.to_csv(os.path.join(ROOT, "viewer", "data", "pred_" + nome + ".csv"), index=False)
            print("gravado evaluation/results/{0}.csv e viewer/data/pred_{0}.csv".format(nome))


if __name__ == "__main__":
    main()
