"""Detector de bola morta v2 x v3 x v4 x v5 (analise/paradas.py, Config() x Config.v3() x .v4() x .v5()).

    python analise/avaliar_v3.py              # tabela v2 x v3, casos (gols, 57942, 20636) e rótulos
    python analise/avaliar_v3.py --exportar   # e grava out/tracking/{D,paradas}_{v2,v3,v4,v5}.csv
    python analise/exportar_viewer_tracking.py                       # -> viewer/data/pred_v3.csv, eventos_v3.json
    python analise/comparar_posse.py --d v3 --exportar posse_viterbi_v3   # posse Viterbi com o D v3
    python analise/comparar_posse.py --d v4 --exportar posse_viterbi_v4   # ... e com o D v4
    python analise/casar_campeao.py --d v5 --min-posse-s 1 --exportar     # D v5 + H/A do campeão

Mesma variante de produção do v2: bola atd_bruto, "+fora" (bola na linha ou além) e "+uniao".
A bola "fundida" entra só no v3, nos sinais de bola (gol, veto de bola viva).

Gabaritos (nenhum treina nada):
    publico   GT público (4 D)
    blocos    os 2 blocos de posse rotulados no viewer (D, H, A)
    manuais   analise/out/rotulos_manuais.csv: erro de início/fim e falso D nos 'viva'
    clipes    analise/out/rotulos_clipes.csv: vereditos do painel "Clipes D" do viewer

Os rótulos à mão (blocos, manuais, clipes) só existem na máquina de quem rotulou e servem só para as
métricas: sem eles as colunas correspondentes ficam vazias, e os D exportados saem iguais.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import comparar_tracking as CT  # noqa: E402
import eventos as E  # noqa: E402
import fontes_bola as B  # noqa: E402
import paradas as P  # noqa: E402
import rotular_paradas as RP  # noqa: E402
import rotulos as R  # noqa: E402
from features import MAX_PLAYER_SPEED, VEL_HALF_WINDOW, centred_velocity, load_atd  # noqa: E402
from features_coletivas import load_all  # noqa: E402

FPS = R.FPS
FONTE = "atd_bruto"
CACHE = os.path.join(HERE, "out", "v3_cache.pkl")
OUT = CT.OUT


def mmss(f):
    s = f / FPS
    return "{:02d}:{:04.1f}".format(int(s // 60), s % 60)


def preparar(cache=True):
    """Sinais, v1, offsets e eventos de início da variante de produção (cache em out/v3_cache.pkl)."""
    if cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as h:
            return pickle.load(h)
    X, meta = load_all()
    I = meta["index"]
    players, _, slot_teams, _ = load_atd()
    vel = centred_velocity(players, VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    gt = R.gt_public()
    cov = np.flatnonzero(gt != "X")
    gt_window = np.zeros(meta["n_frames"], bool)
    gt_window[cov[0] - 60 * FPS:cov[-1] + 60 * FPS] = True
    ball = B.carregar_bola(FONTE)
    Xf = CT.features_da_fonte(X, I, ball, players, vel, slot_teams)
    F, I2 = T.stack(Xf, I, T.context_features(Xf, I))
    S = E.Sinais(Xf, I, F, I2, ball=ball)
    train = (R.weak_labels(Xf, I, use_line=False, alive_rule="jogadores")[0] >= 0) & S.graded & ~gt_window
    v1, _ = CT.arvore_v1(F, I2, Xf, I, train)
    offsets, _ = E.calibrar_offset_smart_tagging(S)
    inicio = E.eventos_inicio(S, fora=True)
    S.F = S.F.astype(np.float32)
    pack = dict(S=S, v1=v1, offsets=offsets, inicio=inicio)
    if cache:
        with open(CACHE, "wb") as h:
            pickle.dump(pack, h)
    return pack


def rodar(pack, cfg):
    return P.detectar_paradas(pack["S"], cfg, inicio=pack["inicio"], offsets=pack["offsets"], v1_dead=pack["v1"])


def metricas_blocos(d, base):
    gt = RP.gt_blocos(len(d))
    cov = gt != "X"
    yD = gt == "D"
    tp, fp, fn = (d & yD)[cov].sum(), (d & ~yD)[cov].sum(), (~d & yD)[cov].sum()
    a = T.evaluate(gt, T.combine_with_baseline(base, d, R.graded_mask(len(d))))
    return dict(blocos_P=tp / max(1, tp + fp), blocos_R=tp / max(1, tp + fn), blocos_macroF1=a["macro_f1"])


def metricas_manuais(d):
    if not os.path.exists(RP.OUT):
        return {}, None
    av = RP.avaliar(RP.carregar(), d)
    viva = av[av.real == "viva"]
    par = av[av.real == "parada"]
    return dict(manuais_falsoD_viva=viva.falso_D_frac.mean(), manuais_cobertura=par.cobertura.mean(),
                manuais_err_ini_s=par.erro_inicio_s.abs().median(), manuais_err_fim_s=par.erro_fim_s.abs().median()), av


PRED_CLIPES = os.path.join(HERE, "..", "viewer", "data", "pred_posse_viterbi_v3.csv")


def _runs_D(d):
    r = np.diff(np.r_[0, d.astype(int), 0])
    return list(zip(np.flatnonzero(r == 1), np.flatnonzero(r == -1)))


def metricas_clipes(d, fonte="pred_posse_viterbi_v3"):
    """Vereditos do painel "Clipes D" (viewer) contra um D qualquer.

    falsoD     fração de D que sobra nos clipes 'falso' (deveria ser 0)
    err_ini/fim  |erro| mediano (s) do início/fim do D que cobre um clipe 'ajustado' (bordas marcadas)
    retido     fração de D mantida nos clipes NÃO marcados até o último clipe revisado: o usuário
               passou por eles sem reclamar, então tomo como certos (sem eles, mudar regra é às cegas)
    """
    df = RP.carregar_clipes()
    if df is None or not len(df):
        return {}
    df = df[df.fonte == fonte]
    out = {"clipes_revisados": len(df)}
    falsos = df[df.veredito == "falso"]
    out["clipes_falsoD"] = float(np.mean([d[int(r.inicio):int(r.fim)].mean() for r in falsos.itertuples()])) if len(falsos) else np.nan
    runs = _runs_D(d)
    e_ini, e_fim = [], []
    for r in df[df.veredito == "ajustado"].itertuples():
        a = int(r.inicio_real) if pd.notna(r.inicio_real) and str(r.inicio_real) != "" else int(r.inicio)
        b = int(r.fim_real) if pd.notna(r.fim_real) and str(r.fim_real) != "" else int(r.fim)
        cob = [(s, e) for s, e in runs if s < b and e > a]
        if not cob:
            e_ini.append(np.nan); e_fim.append(np.nan)
            continue
        if pd.notna(r.inicio_real) and str(r.inicio_real) != "":
            e_ini.append(abs(cob[0][0] - a) / FPS)
        if pd.notna(r.fim_real) and str(r.fim_real) != "":
            e_fim.append(abs(cob[-1][1] - b) / FPS)
    out["clipes_err_ini_s"] = float(np.nanmean(e_ini)) if e_ini else np.nan
    out["clipes_err_fim_s"] = float(np.nanmean(e_fim)) if e_fim else np.nan
    if os.path.exists(PRED_CLIPES):
        lab = pd.read_csv(PRED_CLIPES).label.values == "D"
        ate = int(df.inicio.max())
        marcados = set(df.inicio.astype(int))
        ret = [d[s:e].mean() for s, e in _runs_D(lab) if e - s >= 10 and s <= ate and s not in marcados]
        out["clipes_retido"] = float(np.mean(ret)) if ret else np.nan
    return out


def casos(d, tab):
    """Os casos que motivaram o v3."""
    st = R.set_pieces()
    gols = R.smart_tags("GOALS")
    ko = st[st.type == "Kick Off (After Goal)"].start_frame.values
    rows = []
    for g in gols.itertuples():
        k = ko[ko >= g.start_frame]
        fim = int(k[0]) if len(k) else int(g.end_frame)
        lin = tab[(tab.tipo == "gol") & (tab.fim >= g.start_frame) & (tab.inicio <= fim + 25 * FPS)] if len(tab) and "tipo" in tab else []
        ini = int(lin.inicio.iloc[0]) if len(lin) else None
        rows.append(dict(caso="gol {} {}".format(g.team, mmss(g.start_frame)), inicio_D=ini,
                         cobre_ate_kickoff=float(d[(ini or fim - 20 * FPS):fim].mean()),
                         D_na_janela=float(d[g.start_frame:fim].mean())))
    for f, nome in ((57942, "57942 (gol 2)"), (20636, "20636 (viva)")):
        rows.append(dict(caso=nome, D_em_volta=float(d[f - 5 * FPS:f + 5 * FPS].mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exportar", action="store_true")
    ap.add_argument("--sem-cache", action="store_true")
    args = ap.parse_args()
    t = time.time()
    pack = preparar(cache=not args.sem_cache)
    print("sinais prontos em {:.0f} s".format(time.time() - t))
    S = pack["S"]
    base = T.load_baseline("baseline_smooth125")
    A = CT.Avaliador(S, base)
    rows, avs = [], {}
    for nome, cfg in (("v2", P.Config(v1_uniao=True)), ("v3", P.Config.v3()), ("v4", P.Config.v4()), ("v5", P.Config.v5())):
        d, tab = rodar(pack, cfg)
        m_man, av = metricas_manuais(d)
        rows.append(dict(detector=nome, **A.linha(d, pack["offsets"]), **metricas_blocos(d, base), **m_man,
                         **metricas_clipes(d)))
        avs[nome] = av
        print("\n==", nome)
        print(casos(d, tab).round(3).to_string(index=False))
        if args.exportar:
            os.makedirs(OUT, exist_ok=True)
            tab.to_csv(os.path.join(OUT, "paradas_{}.csv".format(nome)), index=False)
            pd.DataFrame({"frame": np.arange(S.N), "morta": d.astype(int)}).to_csv(
                os.path.join(OUT, "D_{}.csv".format(nome)), index=False)
            gk = tab.attrs.get("goleiro")
            if gk is not None and len(gk):      # v5: D tirados porque o goleiro segurava a bola (time dele)
                gk.to_csv(os.path.join(OUT, "goleiro_{}.csv".format(nome)), index=False)
            print("gravado out/tracking/D_{0}.csv e paradas_{0}.csv".format(nome))
            print("viewer: python analise/exportar_viewer_tracking.py  (pred_{0}.csv, eventos_{0}.json)".format(nome))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    res = pd.DataFrame(rows).set_index("detector").T
    print("\n", res.round(3).to_string())
    if avs["v4"] is not None and avs["v5"] is not None:
        m = avs["v4"].merge(avs["v5"], on=["id", "categoria", "real"], suffixes=("_v4", "_v5"))
        print("\nrótulos manuais (v4 x v5):")
        print(m.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
