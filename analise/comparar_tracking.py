"""Roda o detector de bola morta (árvore v1 e máquina de estados v2) com cada fonte de bola.

    python analise/comparar_tracking.py        # grava analise/out/tracking/

Tudo o que não depende da bola fica igual: áudio, velocidade dos jogadores, Smart Tagging, rótulo
fraco. O que muda por fonte: Sinais.ball, as features de perseguição (nó N3 da árvore) e, nas
variantes "+fora", a regra de bola na linha (eventos_saida_bola(fora=True)). As variantes "+uniao"
ligam Config.v1_uniao: um evento de início dentro de um trecho v1 não corta mais esse trecho.

A linha atd_bin é o teste de regressão: tem de dar o resultado do notebook 03.

Como o GT tem só 4 trechos D, o script também mede contra os set pieces do jogo inteiro, que não
dependem da bola: cobertura por tipo, recall e aproveitamento dos eventos de saída de bola, e um
teste pareado por set piece contra a referência.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import eventos as E  # noqa: E402
import fontes_bola as B  # noqa: E402
import paradas as P  # noqa: E402
import rotulos as R  # noqa: E402
from features import MAX_PLAYER_SPEED, VEL_HALF_WINDOW, centred_velocity, load_atd  # noqa: E402
from features_coletivas import Bag, chase_features, load_all  # noqa: E402

FPS = R.FPS
OUT = os.path.join(HERE, "out", "tracking")
REF = "atd_bin"
NODES = [   # os mesmos do notebook 02
    {"id": "N1", "nome": "futuro lento", "regra": [("speed_fut5s_mean", "<", None)]},
    {"id": "N2", "nome": "lento e desorganizado",
     "regra": [("speed_c5s_mean", "<", None), ("pol_c5s_mean", "<", None)]},
    {"id": "N3", "nome": "bola sumiu na linha, ninguém persegue",
     "regra": [("ball_lost_at_line_s", ">", 1.0), ("closing_c5s_mean", "<", None)]},
    {"id": "N4", "nome": "times afastados e andando",
     "regra": [("teams_centroid_d_m", ">", None), ("a_frac_walk", ">", None)]},
]
COVWIN = {"Corner Kick": 20, "Goal Kick": 15, "Throw-in": 8, "Free Kick": 8, "Kick Off (After Goal)": 20}
SAIDA_TAGS = ("Corner Kick", "Goal Kick", "Throw-in")     # reinícios que a saída de bola deve anunciar
SAIDA_JANELA_S = 45


def features_da_fonte(X, I, ball, players, vel, slot_teams):
    """Cópia de X com as features de perseguição recalculadas para esta bola."""
    bag = Bag(len(players))
    chase_features(bag, players, vel, slot_teams, ball)
    X = X.copy()
    for name, values, *_ in bag.items:
        X[:, I[name]] = values
    return X


def arvore_v1(F, I2, X, I, train):
    yw, _ = R.weak_labels(X, I, use_line=False, alive_rule="jogadores")
    tree = T.RuleTree(NODES, min_precision=0.9, weight=3.0).fit(F, I2, yw.astype(int), train)
    dead, _ = tree.predict(F, I2)
    return T.clean_runs(dead, min_dead_s=3, min_alive_s=2), tree


class Avaliador:
    def __init__(self, S, base125):
        self.gt = R.gt_public()
        self.cov, self.yD = self.gt != "X", self.gt == "D"
        self.g = S.graded
        self.base = base125
        self.gt_team = np.where(np.isin(self.gt, ["H", "A"]), self.gt, base125)
        self.sp = R.set_pieces()

    def por_set_piece(self, d, offsets):
        return pd.Series({i: d[r.start_frame - COVWIN[r.type] * FPS:
                               r.start_frame + int(offsets.get(r.type, 3) * FPS)].mean()
                          for i, r in self.sp.iterrows() if r.type in COVWIN})

    def linha(self, d, offsets):
        tp = (d & self.yD)[self.cov].sum()
        fp = (d & ~self.yD)[self.cov].sum()
        fn = (~d & self.yD)[self.cov].sum()
        a = T.evaluate(self.gt, T.combine_with_baseline(self.base, d, self.g))
        b = T.evaluate(self.gt, T.combine_with_baseline(self.gt_team, d, self.g))
        rep = T.dead_segment_report(d, self.gt)
        cob = self.por_set_piece(d, offsets)
        tipos = self.sp.type.reindex(cob.index)
        out = dict(precisao_D=tp / max(1, tp + fp), recall_D=tp / max(1, tp + fn), macroF1=a["macro_f1"],
                   spellF1_125=a["spell_f1_mean"], spellF1_timeGT=b["spell_f1_mean"],
                   erro_inicio_s=np.nanmedian(np.abs(rep.atraso_inicio_s)), D_no_jogo=d[self.g].mean(),
                   paradas=len(T.runs(d & self.g)), cob_todos=cob.mean())
        out.update({"cob_" + t.split()[0].lower(): v for t, v in cob.groupby(tipos).mean().items()})
        return out

    def saidas(self, inicio):
        """Eventos de saída de bola contra os reinícios marcados no jogo inteiro."""
        fr = np.array([e["frame"] for e in inicio if e["fonte"] == "saida_bola"])
        tag = self.sp[self.sp.type.isin(SAIDA_TAGS)].start_frame.values
        j = SAIDA_JANELA_S * FPS
        antes = [(s - fr[(fr >= s - j) & (fr <= s)].max()) / FPS for s in tag if ((fr >= s - j) & (fr <= s)).any()]
        util = np.mean([((tag >= f) & (tag <= f + j)).any() for f in fr]) if len(fr) else np.nan
        return dict(saidas=len(fr), saida_recall=len(antes) / len(tag), saida_util=util,
                    saida_antecede_s=float(np.median(antes)) if antes else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    X, meta = load_all()
    I, N = meta["index"], meta["n_frames"]
    players, _, slot_teams, _ = load_atd()
    vel = centred_velocity(players, VEL_HALF_WINDOW, MAX_PLAYER_SPEED)
    gt = R.gt_public()
    cov = np.flatnonzero(gt != "X")
    gt_window = np.zeros(N, bool)
    gt_window[cov[0] - 60 * FPS:cov[-1] + 60 * FPS] = True
    base125 = T.load_baseline("baseline_smooth125")

    rows, cobs, nos = [], {}, []
    for fonte in B.FONTES:
        print("==", fonte, flush=True)
        ball = B.carregar_bola(fonte)
        Xf = X if fonte == REF else features_da_fonte(X, I, ball, players, vel, slot_teams)
        F, I2 = T.stack(Xf, I, T.context_features(Xf, I))
        S = E.Sinais(Xf, I, F, I2, ball=ball)
        if fonte == "fundida":
            S.ball_qualidade, S.ball_raio = B.extras_fundida()
        A = Avaliador(S, base125)
        cobertura = np.mean(~np.isnan(ball[S.graded, 0]))
        train = (R.weak_labels(Xf, I, use_line=False, alive_rule="jogadores")[0] >= 0) & S.graded & ~gt_window
        v1, tree = arvore_v1(F, I2, Xf, I, train)
        offsets, _ = E.calibrar_offset_smart_tagging(S)
        desc = tree.describe()
        desc.insert(0, "fonte", fonte)
        nos.append(desc)
        rows.append(dict(fonte=fonte, detector="v1", cobertura_bola=cobertura, **A.linha(v1, offsets)))
        cobs[fonte + " v1"] = A.por_set_piece(v1, offsets)
        for fora, uniao in ((False, False), (False, True), (True, False), (True, True)):
            if fora and fonte == REF:
                continue                       # no .bin a bola nunca passa da linha
            inicio = E.eventos_inicio(S, fora=fora)
            d, tab = P.detectar_paradas(S, P.Config(v1_uniao=uniao), inicio=inicio, offsets=offsets, v1_dead=v1)
            nome = fonte + (" +fora" if fora else "") + (" +uniao" if uniao else "")
            rows.append(dict(fonte=nome, detector="v2", cobertura_bola=cobertura,
                             **A.linha(d, offsets), **A.saidas(inicio)))
            cobs[nome] = A.por_set_piece(d, offsets)
            arq = nome.replace(" +", "_")
            tab.to_csv(os.path.join(OUT, "paradas_{}.csv".format(arq)), index=False)
            pd.DataFrame({"frame": np.arange(N), "morta": d.astype(int)}).to_csv(
                os.path.join(OUT, "D_{}.csv".format(arq)), index=False)

    res = pd.DataFrame(rows)
    ref = cobs[REF]
    pareado = []
    for nome, c in cobs.items():
        if nome == REF or nome.endswith(" v1"):
            continue
        dif = (c - ref).values
        nz = dif[np.abs(dif) > 1e-9]
        pareado.append(dict(fonte=nome, sobe=int((nz > 0).sum()), desce=int((nz < 0).sum()),
                            delta_medio=dif.mean(), p_wilcoxon=wilcoxon(nz).pvalue if len(nz) else np.nan))
    par = pd.DataFrame(pareado)
    res.to_csv(os.path.join(OUT, "comparacao.csv"), index=False)
    par.to_csv(os.path.join(OUT, "pareado_set_pieces.csv"), index=False)
    pd.concat(nos).to_csv(os.path.join(OUT, "nos_arvore.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res.round(3).to_string(index=False))
    print(par.round(4).to_string(index=False))
    print(pd.concat(nos)[["fonte", "no", "regra", "precisao_treino", "podado"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
