"""Quanto o dono do reinício (analise/dono_reinicio.py) melhora o spellF1, medido em OUTROS jogos.

    METRICA_DIR=/caminho/para/sample-data python validacao_metrica/testar_dono_reinicio.py

Usa os jogos de amostra 1 e 2 da Metrica (validacao_metrica/metrica.py) e o scorer oficial
(evaluation/possession_metrics.py). O H/A vem do baseline oficial (jogador mais perto da bola, voto
majoritário de ±50 frames). As paradas são as do gabarito ("D exato"), para isolar o efeito do dono;
uma variante desloca o início e o fim de cada parada em até ±2 s, como um detector imperfeito.

Variantes, todas com o mesmo D:
    V1 baseline + D            os H/A do baseline com o D por cima (arvore.combine_with_baseline),
                               como o notebook 03 faz hoje
    V2 + máscara               o mesmo, mas a evidência do baseline DENTRO das paradas é apagada
                               antes da suavização
    V3 + dono estimado         V2 e o dono estimado nos primeiros SEGURAR_S depois de cada reinício
    V4 + dono do gabarito      V2 com o dono verdadeiro: o teto deste mecanismo

Cenários de bola: "bola Metrica" (a bola some em ~7% dos frames de jogo vivo) e "bola apagada"
(lacunas aleatórias até ~35% dos frames de jogo vivo, mais perto do ATD do desafio, que não vê a
bola em 41% dos frames avaliados). Cada cenário com lacunas é a média de 5 sementes.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "analise"), os.path.join(ROOT, "evaluation"),
          os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)

import metrica as M  # noqa: E402
import dono_reinicio as DR  # noqa: E402
import possession_metrics as pm  # noqa: E402
from arvore import combine_with_baseline  # noqa: E402

FPS = 25
JANELA = 50                 # baseline_smooth50
SEMENTES = range(5)
OUT = os.path.join(HERE, "out")


def smooth(labels, window=JANELA):
    """O voto majoritário de starter/baseline.smooth, sobre rótulos H/A/D (D = ninguém)."""
    h = np.concatenate(([0], np.cumsum(labels == "H")))
    a = np.concatenate(([0], np.cumsum(labels == "A")))
    i = np.arange(len(labels))
    lo, hi = np.maximum(0, i - window), np.minimum(len(labels), i + window + 1)
    nh, na = h[hi] - h[lo], a[hi] - a[lo]
    out = np.full(len(labels), "D", "<U1")
    seen = nh + na > 0
    out[seen] = np.where(nh[seen] >= na[seen], "H", "A")
    return out


def mais_proximo(J, ball):
    """Evidência por frame do baseline: time do jogador mais perto da bola, D sem bola."""
    out = np.full(J.N, "D", "<U1")
    f, t, _ = DR._proximos(J.players, J.time_slot, ball, 0, J.N)
    out[f] = t
    return out


def degradar(ball, vivo, alvo=0.35, media_s=1.5, seed=0):
    """Apaga a bola em lacunas aleatórias até `alvo` dos frames de jogo vivo ficarem sem bola."""
    rng = np.random.default_rng(seed)
    b = ball.copy()
    n_vivo = vivo.sum()
    while np.isnan(b[vivo, 0]).mean() < alvo:
        s = rng.integers(0, len(b))
        e = s + max(1, int(rng.exponential(media_s * FPS)))
        b[s:e] = np.nan
    return b


def deslocar(paradas, seed, max_s=2.0):
    rng = np.random.default_rng(100 + seed)
    p = paradas.copy()
    k = int(max_s * FPS)
    p["inicio"] = p.inicio + rng.integers(-k, k + 1, len(p))
    p["fim"] = p.fim + rng.integers(-k, k + 1, len(p))
    p["fim"] = np.maximum(p.fim, p.inicio + FPS)
    return p


def mascara_d(paradas, n):
    d = np.zeros(n, bool)
    for r in paradas.itertuples():
        d[max(0, r.inicio):r.fim] = True
    return d


def pontuar(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
    return r["spell_f1_mean"], r["macro_f1"]


def rodar(J, ball, paradas, segurar_s=DR.SEGURAR_S, tipo_do_gabarito=False):
    """spellF1 e macroF1 de V1..V4 e a acurácia do dono estimado, para uma bola e um conjunto de paradas."""
    graded = J.gt != "X"
    dead = mascara_d(paradas, J.N) & graded
    raw = mais_proximo(J, ball)
    base = smooth(raw)
    res = {}
    res["V1 baseline + D"] = pontuar(J.gt, combine_with_baseline(base, dead, graded))

    sem_dono = paradas[["inicio", "fim"]].assign(dono=None)
    res["V2 + máscara"] = pontuar(J.gt, combine_with_baseline(
        smooth(DR.aplicar_donos(raw, sem_dono, segurar_s)), dead, graded))

    entrada = paradas[["inicio", "fim"] + (["tipo"] if tipo_do_gabarito else [])]
    donos = DR.estimar_donos(entrada, J.players, J.time_slot, ball, rotulos=base)
    res["V3 + dono estimado"] = pontuar(J.gt, combine_with_baseline(
        smooth(DR.aplicar_donos(raw, donos, segurar_s)), dead, graded))

    verdade = paradas[["inicio", "fim"]].assign(dono=J.reinicios.dono.values)
    res["V4 + dono do gabarito"] = pontuar(J.gt, combine_with_baseline(
        smooth(DR.aplicar_donos(raw, verdade, segurar_s)), dead, graded))

    certo = donos.dono.values == J.reinicios.dono.values
    return res, donos.assign(verdade=J.reinicios.dono.values, certo=certo)


def acuracia_sinais(donos):
    rows = []
    for s in ("tag", "cobrador", "ultimo_toque", "posse_antes"):
        v = donos["voto_" + s]
        tem = v.isin(["H", "A"])
        rows.append(dict(sinal=s, cobertura=tem.mean(),
                         acerto=(v[tem] == donos.verdade[tem]).mean() if tem.any() else np.nan))
    for t, g in donos.groupby("tipo_dono"):
        for s in ("cobrador", "ultimo_toque", "posse_antes"):
            v = g["voto_" + s]
            tem = v.isin(["H", "A"])
            rows.append(dict(sinal="{} | {}".format(s, t), cobertura=tem.mean(),
                             acerto=(v[tem] == g.verdade[tem]).mean() if tem.any() else np.nan))
    rows.append(dict(sinal="combinado", cobertura=donos.dono.isin(["H", "A"]).mean(), acerto=donos.certo.mean()))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    tabelas, sinais = [], []
    for g in (1, 2):
        J = M.carregar_jogo(g)
        vivo = np.isin(J.gt, ["H", "A"])
        P = J.reinicios[["inicio", "fim", "tipo"]]
        cenarios = [("bola Metrica, D exato", [(J.ball, P)])]
        cenarios.append(("bola apagada, D exato",
                         [(degradar(J.ball, vivo, seed=s), P) for s in SEMENTES]))
        cenarios.append(("bola apagada, D ±2 s",
                         [(degradar(J.ball, vivo, seed=s), deslocar(P, s)) for s in SEMENTES]))
        for nome, casos in cenarios:
            acc = []
            for ball, p in casos:
                res, donos = rodar(J, ball, p)
                acc.append(res)
                if p is P:
                    sinais.append(acuracia_sinais(donos).assign(jogo=g, cenario=nome))
            for v in acc[0]:
                sf = np.mean([a[v][0] for a in acc]); mf = np.mean([a[v][1] for a in acc])
                tabelas.append(dict(jogo=g, cenario=nome, variante=v, spellF1=sf, macroF1=mf))
            print("jogo {} | {:<24} ".format(g, nome)
                  + " | ".join("{} {:.3f}".format(v.split()[0], np.mean([a[v][0] for a in acc])) for v in acc[0]))
        # sensibilidade ao tempo em que o dono vale
        for seg in (0.5, 1.0, 2.0, 4.0):
            acc = [rodar(J, degradar(J.ball, vivo, seed=s), P, segurar_s=seg)[0] for s in SEMENTES]
            sf = np.mean([a["V3 + dono estimado"][0] for a in acc])
            tabelas.append(dict(jogo=g, cenario="segurar {} s (bola apagada)".format(seg),
                                variante="V3 + dono estimado", spellF1=sf,
                                macroF1=np.mean([a["V3 + dono estimado"][1] for a in acc])))
            print("jogo {} | segurar {:>3} s -> V3 spellF1 {:.3f}".format(g, seg, sf))
    T = pd.DataFrame(tabelas)
    S = pd.concat(sinais)
    T.to_csv(os.path.join(OUT, "resultados.csv"), index=False)
    S.to_csv(os.path.join(OUT, "acuracia_sinais.csv"), index=False)
    print()
    print(T.pivot_table(index=["cenario", "variante"], columns="jogo", values="spellF1").round(3).to_string())
    print()
    print(S.pivot_table(index=["cenario", "sinal"], columns="jogo", values=["cobertura", "acerto"]).round(2).to_string())


if __name__ == "__main__":
    main()
