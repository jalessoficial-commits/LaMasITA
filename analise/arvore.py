"""Árvore determinística de bola morta: contexto temporal, cascata de regras, suavização e avaliação.

    from arvore import *       # com analise/ no sys.path (o notebook 02 mostra o uso completo)

Peças, na ordem em que um frame passa por elas:

    context_features(X, I)    janelas centradas e FUTURAS (a submissão é offline). O notebook 01 mostrou que o
                              apito é gradual (a velocidade cai ao longo de ~10 s) e o reinício é abrupto, então
                              olhar só o passado marca o início do D com atraso.
    RuleTree                  cascata de nós legíveis: cada frame para no primeiro nó que dispara e
                              fica com o id desse nó; os limiares são ajustados no rótulo fraco.
    hysteresis / clean_runs   tiram o "pisca-pisca" por frame: histerese e durações mínimas.
    viterbi2                  alternativa: HMM de 2 estados com durações médias explícitas.
    combine_with_baseline     põe o D por cima dos H/A de uma submissão (ex.: baseline_smooth50).
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "evaluation"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

FPS = 25


# ---- contexto temporal --------------------------------------------------------------------------

def _roll(x, w, how, where):
    s = pd.Series(x)
    if where == "c":
        r = s.rolling(w, center=True, min_periods=max(1, w // 4))
        return getattr(r, how)().values
    if where == "f":                          # janela [t, t + w)
        r = s[::-1].rolling(w, min_periods=max(1, w // 4))
        return getattr(r, how)().values[::-1]
    r = s.rolling(w, min_periods=max(1, w // 4))  # passado: (t - w, t]
    return getattr(r, how)().values


def context_features(X, I):
    """dict nome -> array (n,) com janelas centradas e futuras dos sinais-chave."""
    speed = X[:, I["all_speed_mean"]]
    pol = X[:, I["all_polarization"]]
    chase = X[:, I["chase_cos_k3_all"]]
    closing = X[:, I["frac_closing"]]
    out = {}
    for w_s in (3, 5, 10):
        w = int(w_s * FPS)
        out["speed_c{}s_mean".format(w_s)] = _roll(speed, w, "mean", "c")
        out["speed_c{}s_max".format(w_s)] = _roll(speed, w, "max", "c")
        out["pol_c{}s_mean".format(w_s)] = _roll(pol, w, "mean", "c")
        out["chase_c{}s_mean".format(w_s)] = _roll(chase, w, "mean", "c")
        out["closing_c{}s_mean".format(w_s)] = _roll(closing, w, "mean", "c")
    for w_s in (3, 5):
        w = int(w_s * FPS)
        out["speed_fut{}s_mean".format(w_s)] = _roll(speed, w, "mean", "f")
        out["speed_fut{}s_max".format(w_s)] = _roll(speed, w, "max", "f")
        out["speed_past{}s_max_".format(w_s)] = _roll(speed, w, "max", "p")
        # "cercado de lentidão": baixo só se o passado E o futuro próximos forem lentos
        out["enc_speed_max_{}s".format(w_s)] = np.fmax(out["speed_past{}s_max_".format(w_s)],
                                                       out["speed_fut{}s_max".format(w_s)])
        out["enc_speed_min_{}s".format(w_s)] = np.fmin(out["speed_past{}s_max_".format(w_s)],
                                                       out["speed_fut{}s_max".format(w_s)])
    # aceleração coletiva futura: o reinício é um pico de aceleração logo depois do mínimo
    s1 = _roll(speed, FPS, "mean", "c")
    out["speed_gain_fut3s"] = _roll(s1, 3 * FPS, "max", "f") - s1
    return {k: v.astype(np.float32) for k, v in out.items()}


def stack(X, I, ctx):
    """X e o contexto lado a lado, com índice único."""
    names = list(ctx)
    I2 = dict(I)
    for j, n in enumerate(names):
        I2[n] = X.shape[1] + j
    return np.hstack([X, np.stack([ctx[n] for n in names], 1)]), I2


# ---- limiares -----------------------------------------------------------------------------------

def fit_threshold(x, y, direction, min_precision=0.9, weight=None):
    """Limiar que maximiza o recall de y=1 com precisão >= min_precision.

    direction '<': dispara quando x < t; '>': quando x > t. weight reequilibra as classes (o rótulo
    fraco tem muito mais 'viva' do que 'morta' de verdade num jogo; sem peso, a precisão sai otimista).
    Retorna (t, precisão, recall); se nenhum limiar atinge a precisão, usa o de melhor F1.
    """
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    w = np.ones(len(y)) if weight is None else np.where(y == 1, weight, 1.0)
    qs = np.unique(np.nanpercentile(x, np.linspace(1, 99, 197)))
    best, best_f1 = None, None
    for t in qs:
        fire = x < t if direction == "<" else x > t
        tp, fp = np.sum(w * (fire & (y == 1))), np.sum(w * (fire & (y == 0)))
        fn = np.sum(w * (~fire & (y == 1)))
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if prec >= min_precision and (best is None or rec > best[2]):
            best = (float(t), float(prec), float(rec))
        if best_f1 is None or f1 > best_f1[0]:
            best_f1 = (f1, float(t), float(prec), float(rec))
    return best if best is not None else best_f1[1:]


class RuleTree:
    """Cascata determinística. nodes = lista de dicts:

        {"id": "N1", "nome": "...", "regra": [(feature, '<' | '>', limiar | None), ...], "saida": "D"}

    Um frame dispara um nó quando TODAS as condições da regra são verdadeiras (NaN = falso). O
    primeiro nó que dispara decide; se nenhum dispara, o frame é viva (nó "N_viva"). Limiar None é
    ajustado por fit() no rótulo fraco, nó a nó, só sobre os frames que chegaram até aquele nó.
    """

    def __init__(self, nodes, min_precision=0.9, weight=None, prune_below=0.7):
        self.nodes = [dict(n, regra=[list(c) for c in n["regra"]]) for n in nodes]
        self.min_precision, self.weight, self.prune_below = min_precision, weight, prune_below

    def _fires(self, node, F, I):
        out = np.ones(len(F), bool)
        for feat, op, t in node["regra"]:
            x = F[:, I[feat]]
            out &= (x < t) if op == "<" else (x > t)
        return out

    def fit(self, F, I, y, train):
        remaining = train.copy()
        for node in self.nodes:
            fires_fixed = np.ones(len(F), bool)
            for c in node["regra"]:
                if c[2] is not None:
                    x = F[:, I[c[0]]]
                    fires_fixed &= (x < c[2]) if c[1] == "<" else (x > c[2])
            for c in node["regra"]:
                if c[2] is None:
                    sel = remaining & fires_fixed
                    t, prec, rec = fit_threshold(F[sel, I[c[0]]], y[sel], c[1], self.min_precision, self.weight)
                    c[2] = t
                    x = F[:, I[c[0]]]
                    fires_fixed &= (x < t) if c[1] == "<" else (x > t)
            fired = self._fires(node, F, I)
            hit = remaining & fired
            w = np.where(y[hit] == 1, self.weight or 1.0, 1.0)
            node["treino_precisao"] = float(np.sum(w * (y[hit] == 1)) / np.sum(w)) if hit.any() else np.nan
            node["treino_cobertura_D"] = float(np.sum(hit & (y == 1)) / max(1, np.sum(train & (y == 1))))
            # poda: um nó que não chega perto da precisão alvo no treino só gera falso D
            node["podado"] = bool(not (node["treino_precisao"] >= self.prune_below))
            if not node["podado"]:
                remaining &= ~fired
        return self

    def predict(self, F, I):
        """(is_dead bool, node_id str) por frame."""
        node = np.full(len(F), "N_viva", dtype="<U8")
        dead = np.zeros(len(F), bool)
        free = np.ones(len(F), bool)
        for n in self.nodes:
            if n.get("podado"):
                continue
            fired = free & self._fires(n, F, I)
            node[fired] = n["id"]
            dead[fired] = n.get("saida", "D") == "D"
            free &= ~fired
        return dead, node

    def describe(self):
        rows = []
        for n in self.nodes:
            cond = " E ".join("{} {} {:.3g}".format(f, op, t) for f, op, t in n["regra"])
            rows.append(dict(no=n["id"], nome=n["nome"], regra=cond, saida=n.get("saida", "D"),
                             precisao_treino=n.get("treino_precisao"), cobertura_D_treino=n.get("treino_cobertura_D"),
                             podado=n.get("podado", False)))
        return pd.DataFrame(rows)


# ---- suavização temporal --------------------------------------------------------------------------

def runs(mask):
    padded = np.concatenate(([False], np.asarray(mask, bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return list(zip(starts, ends))


def clean_runs(dead, min_dead_s=2.0, min_alive_s=1.0):
    """Preenche buracos de viva < min_alive_s dentro de D e depois apaga D < min_dead_s."""
    d = dead.copy()
    for s, e in runs(~d):
        if e - s < min_alive_s * FPS and s > 0 and e < len(d):
            d[s:e] = True
    for s, e in runs(d):
        if e - s < min_dead_s * FPS:
            d[s:e] = False
    return d


def hysteresis(score, t_on, t_off):
    """Liga quando score > t_on, desliga quando score < t_off (t_off <= t_on). NaN mantém o estado."""
    out = np.zeros(len(score), bool)
    state = False
    on = score > t_on
    off = score < t_off
    # vetorizado: o estado muda só nos frames on/off; propaga o último evento
    ev = np.where(on, 1, np.where(off, 0, -1))
    idx = np.where(ev >= 0, np.arange(len(ev)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.where(idx >= 0, ev[np.maximum(idx, 0)] == 1, state)
    return out


def smooth_score(score, w_s):
    return pd.Series(score).rolling(max(1, int(w_s * FPS)), center=True, min_periods=1).mean().values


def viterbi2(p_dead, mean_dead_s=15.0, mean_alive_s=45.0, prior_dead=0.2, eps=1e-3):
    """HMM de 2 estados (0 viva, 1 morta). Emissão: p_dead já é a probabilidade a posteriori por
    frame; divide pelo prior para virar verossimilhança. As transições vêm das durações médias:
    P(ficar) = 1 - 1 / (duração * FPS)."""
    p = np.clip(np.nan_to_num(p_dead, nan=prior_dead), eps, 1 - eps)
    le = np.stack([np.log(1 - p) - np.log(1 - prior_dead), np.log(p) - np.log(prior_dead)], 1)
    sa, sd = 1 - 1 / (mean_alive_s * FPS), 1 - 1 / (mean_dead_s * FPS)
    lt = np.log(np.array([[sa, 1 - sa], [1 - sd, sd]]))
    n = len(p)
    back = np.zeros((n, 2), np.int8)
    v = np.log([1 - prior_dead, prior_dead]) + le[0]
    for t in range(1, n):
        cand = v[:, None] + lt              # de i para j
        back[t] = np.argmax(cand, 0)
        v = cand[back[t], [0, 1]] + le[t]
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(v))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(bool)


# ---- avaliação ----------------------------------------------------------------------------------

def load_baseline(name="baseline_smooth50"):
    import possession_metrics as pm
    return pm.read_label_csv(os.path.join(ROOT, "evaluation", "results", name + ".csv"))["label"].values.astype("<U1")


def combine_with_baseline(base, dead, graded=None):
    """H/A do baseline, com o D trocado pelo nosso. Onde o baseline dizia D e nós dizemos viva, herda o
    último H/A do baseline (forward fill) -- o baseline só diz D quando a bola do ATD some."""
    lab = base.copy()
    team = np.where(np.isin(lab, ["H", "A"]), lab, "")
    idx = np.where(team != "", np.arange(len(lab)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = team[idx]
    filled[filled == ""] = "H"
    lab = np.where(dead, "D", filled)
    if graded is not None:
        lab = np.where(graded, lab, "X")
    return lab


def evaluate(gt_labels, pred_labels):
    import possession_metrics as pm
    r = pm.evaluate(pm.label_codes(gt_labels), pm.label_codes(pred_labels))
    keys = ["spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells", "n_pred_spells", "fragmentation"]
    return {k: r[k] for k in keys}


def dead_segment_report(dead, gt_labels):
    """IoU de cada trecho D do GT com o trecho previsto que mais o cobre, e o atraso no início e no fim."""
    rows = []
    pred = runs(dead)
    for s, e in runs(gt_labels == "D"):
        best = None
        for ps, pe in pred:
            inter = min(e, pe) - max(s, ps)
            if inter > 0:
                iou = inter / (max(e, pe) - min(s, ps))
                if best is None or iou > best[0]:
                    best = (iou, ps, pe)
        if best:
            rows.append(dict(D_inicio=s, D_fim=e, dur_s=(e - s) / FPS, iou=round(best[0], 2),
                             atraso_inicio_s=(best[1] - s) / FPS, atraso_fim_s=(best[2] - e) / FPS))
        else:
            rows.append(dict(D_inicio=s, D_fim=e, dur_s=(e - s) / FPS, iou=0.0, atraso_inicio_s=np.nan, atraso_fim_s=np.nan))
    return pd.DataFrame(rows)
