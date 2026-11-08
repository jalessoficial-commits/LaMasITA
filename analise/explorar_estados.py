"""Exploração: os frames formam "estados" de movimento distintos? Bola morta cai num deles?

    python viewer/features.py               # uma vez: gera viewer/data/features_tactical.*
    python analise/explorar_estados.py      # ~2-4 min; figuras e tabelas em analise/out/

Cada frame avaliado é uma instância, descrita pelas meta-características de viewer/features.py
(cinemática coletiva, vetor velocidade, forma dos times, relação com a bola). Nenhuma posição crua
entra: posição separa fase tática e região do campo, não bola viva × morta.

O que sai em analise/out/ (fora do git: é derivado dos dados restritos):

    pca_variancia.png, pca_cargas.csv         quanto cada componente explica e o que pesa nela
    proj_<metodo>_<cor>.png                   PCA e t-SNE (e UMAP, se instalado) em R², coloridos
                                              por cluster, GT público, set piece, bola visível e tempo
    gmm_bic.png                               BIC do GMM para k = 2..12
    clusters_perfil.png / .csv                z-score médio de cada feature por cluster
    clusters_resumo.csv                       por cluster: % do jogo, % de D no GT, % perto de set
                                              piece, duração mediana das sequências
    separabilidade.csv / .png                 AUC univariada viva × morta (GT público) e
                                              pré-reinício × jogo corrido (Smart Tagging)
    clusters_por_frame.csv                    frame,cluster para todos os frames avaliados
    relatorio.md                              resumo dos números acima

Dois conjuntos de features: "jogadores" (só movimento, sem bola: sem NaN, cobre todos os frames) e
"jogadores+bola" (inclui as features da bola, só nos frames com bola visível). O primeiro responde a
pergunta direta -- o movimento dos jogadores sozinho separa estados? -- e é o padrão.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "viewer"))
from games import GAME, half_video_blocks  # noqa: E402
from features import load_features  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Params
OUT_DIR = os.path.join(HERE, "out")
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
FPS = 25
STEP = 5                  # 1 frame a cada 0,2 s: ~32 mil instâncias
EMBED_POINTS = 12000      # t-SNE/UMAP numa sub-amostra (custo quadrático-ish)
K_RANGE = range(2, 13)
PCA_VARIANCE = 0.95       # clusters no espaço PCA que retém 95% da variância
SP_BEFORE_S, SP_AFTER_S = 10, 5   # janela "perto de reinício" em torno do início do set piece
FAR_FROM_SP_S = 30        # "jogo corrido": a mais de 30 s de qualquer set piece
SEED = 0

PLAYER_FEATURES = [
    # cinemática
    "all_speed_mean", "all_speed_median", "all_speed_p90", "all_frac_walk", "all_frac_run",
    "h_speed_mean", "a_speed_mean", "h_frac_walk", "a_frac_walk",
    "all_speed_mean_3s", "all_speed_accel", "all_speed_std_2s",
    # vetor velocidade
    "all_polarization", "h_polarization", "a_polarization", "all_milling", "h_milling", "a_milling",
    "all_centroid_speed", "h_centroid_speed", "a_centroid_speed", "teams_centroid_vel_cos",
    "all_centroid_pe_2s",
    # forma
    "all_stretch", "h_stretch", "a_stretch", "all_length", "all_width", "h_length", "a_length",
    "h_width", "a_width", "all_hull_area", "h_hull_area", "a_hull_area", "teams_centroid_d_m",
    "all_frac_own_half", "all_players_in_boxes",
]
BALL_FEATURES = [
    "ball_out_dist_m", "ball_touchline_d_m", "ball_goalline_d_m", "ball_center_d_m",
    "ball_corner_d_m", "ball_speed_1s", "ball_nearest_player_d_m", "ball_players_5m",
    "ball_players_10m", "ball_wcentroid_d_m",
]


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def set_piece_labels(X, index, n_frames):
    """+1 perto de um reinício, 0 em jogo corrido longe de qualquer reinício, -1 no resto.

    Usa as colunas st_* de features.py, que vêm do Smart Tagging. É um rótulo fraco: o início do
    intervalo do set piece fica perto da cobrança, e os segundos antes dela são quase sempre bola
    morta; longe de qualquer reinício é quase sempre bola viva.
    """
    to_next = X[:, index["st_to_next_sp_s"]]
    since = X[:, index["st_since_sp_s"]]
    near = (to_next <= SP_BEFORE_S) | (since <= SP_AFTER_S)
    far = (np.nan_to_num(to_next, nan=1e9) > FAR_FROM_SP_S) & (np.nan_to_num(since, nan=1e9) > FAR_FROM_SP_S)
    out = np.full(n_frames, -1, dtype=np.int8)
    out[far] = 0
    out[near] = 1
    return out


def scatter(xy, colour, title, path, categorical=True, labels=None, cmap="tab10", order=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if categorical:
        values = order if order is not None else sorted(np.unique(colour))
        palette = plt.get_cmap(cmap)
        for i, v in enumerate(values):
            sel = colour == v
            if not sel.any():
                continue
            name = labels.get(v, str(v)) if labels else str(v)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=3, alpha=0.6, color=palette(i % 10),
                       label="{} ({})".format(name, sel.sum()), rasterized=True)
        ax.legend(markerscale=4, fontsize=8, loc="best")
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=colour, s=3, alpha=0.6, cmap=cmap, rasterized=True)
        fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def auc_table(X, index, names, y, mask):
    rows = []
    for name in names:
        v = X[mask, index[name]]
        t = y[mask]
        ok = ~np.isnan(v)
        if ok.sum() < 50 or len(np.unique(t[ok])) < 2 or np.nanstd(v[ok]) == 0:
            rows.append((name, np.nan, "", np.nan, np.nan, ok.mean()))
            continue
        auc = roc_auc_score(t[ok], v[ok])
        rows.append((name, max(auc, 1 - auc), "<" if auc < 0.5 else ">",
                     float(np.median(v[ok][t[ok] == 1])), float(np.median(v[ok][t[ok] == 0])),
                     ok.mean()))
    return rows


def run_lengths_median(labels):
    """Duração mediana (s) das sequências contíguas de cada rótulo."""
    change = np.flatnonzero(np.diff(labels) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    out = {}
    for s, e in zip(starts, ends):
        out.setdefault(labels[s], []).append((e - s) / FPS)
    return {k: float(np.median(v)) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--with-ball", action="store_true",
                        help="inclui as features da bola (só frames com bola visível)")
    parser.add_argument("--k", type=int, default=None, help="força o nº de clusters do GMM")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    X, meta = load_features()
    index = meta["index"]
    n_frames = meta["n_frames"]
    names = PLAYER_FEATURES + (BALL_FEATURES if args.with_ball else [])
    tag = "jogadores+bola" if args.with_ball else "jogadores"

    graded = graded_mask(n_frames)
    gt = pd.read_csv(GT_PATH)["label"].values
    sp = set_piece_labels(X, index, n_frames)

    # ---- instâncias ---------------------------------------------------------------------------
    frames_all = np.flatnonzero(graded)
    F_all = X[frames_all][:, [index[n] for n in names]]
    complete_all = ~np.isnan(F_all).any(axis=1)
    frames = frames_all[complete_all][::STEP]
    F = X[frames][:, [index[n] for n in names]]
    print("{} features ({}), {} instâncias (1 a cada {} frames, completas)".format(
        len(names), tag, len(frames), STEP))

    scaler = StandardScaler().fit(F)
    Z = scaler.transform(F)

    # ---- PCA ----------------------------------------------------------------------------------
    pca = PCA(random_state=SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cum, PCA_VARIANCE) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, len(cum) + 1), pca.explained_variance_ratio_, label="por componente")
    ax.plot(np.arange(1, len(cum) + 1), cum, "k.-", label="acumulada")
    ax.axhline(PCA_VARIANCE, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("componente")
    ax.set_ylabel("variância explicada")
    ax.set_title("PCA ({}): {} componentes para {:.0%}".format(tag, n_keep, PCA_VARIANCE))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pca_variancia.png"), dpi=130)
    plt.close(fig)
    loadings = pd.DataFrame(pca.components_[:6].T, index=names,
                            columns=["PC{}".format(i + 1) for i in range(6)])
    loadings.to_csv(os.path.join(OUT_DIR, "pca_cargas.csv"), float_format="%.3f")
    P = pca.transform(Z)
    P_keep = P[:, :n_keep]
    print("PCA: PC1 {:.1%}, PC2 {:.1%}; {} componentes para {:.0%}".format(
        pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1], n_keep, PCA_VARIANCE))

    # ---- nº de clusters -----------------------------------------------------------------------
    bics = []
    for k in K_RANGE:
        gmm = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(P_keep)
        bics.append(gmm.bic(P_keep))
        print("  GMM k={:2d}  BIC {:.0f}".format(k, bics[-1]))
    bics = np.array(bics)
    # o BIC costuma cair sempre com 30 mil pontos; o "cotovelo" é onde o ganho relativo fica < 1%
    gains = -np.diff(bics) / np.abs(bics[:-1])
    elbow = int(list(K_RANGE)[np.argmax(gains < 0.01)] if (gains < 0.01).any() else list(K_RANGE)[-1])
    k_best = args.k or elbow
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_RANGE), bics, "o-")
    ax.axvline(k_best, color="red", ls="--", lw=0.8, label="k escolhido = {}".format(k_best))
    ax.set_xlabel("k")
    ax.set_ylabel("BIC (menor é melhor)")
    ax.set_title("GMM no espaço PCA ({})".format(tag))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "gmm_bic.png"), dpi=130)
    plt.close(fig)

    gmm = GaussianMixture(k_best, covariance_type="full", random_state=SEED, n_init=3).fit(P_keep)
    cluster = gmm.predict(P_keep)

    hdb_labels = None
    try:
        from sklearn.cluster import HDBSCAN
        hdb_labels = HDBSCAN(min_cluster_size=300, min_samples=30).fit_predict(P_keep)
        n_hdb = len(set(hdb_labels) - {-1})
        print("HDBSCAN: {} clusters, {:.1%} ruído".format(n_hdb, np.mean(hdb_labels == -1)))
    except ImportError:
        n_hdb = None

    # ---- rótulos para colorir -----------------------------------------------------------------
    gt_s = gt[frames]
    sp_s = sp[frames]
    ball_s = X[frames, index["ball_present"]]
    half_s = X[frames, index["half"]]

    # ---- projeções em R² ----------------------------------------------------------------------
    sub = np.sort(rng.choice(len(frames), min(EMBED_POINTS, len(frames)), replace=False))
    # garante que os frames do GT público entram na sub-amostra
    sub = np.union1d(sub, np.flatnonzero(gt_s != "X"))
    embeddings = {"pca": P[:, :2]}
    print("t-SNE em {} pontos ...".format(len(sub)))
    tsne = np.full((len(frames), 2), np.nan)
    tsne[sub] = TSNE(perplexity=40, init="pca", random_state=SEED).fit_transform(P_keep[sub])
    embeddings["tsne"] = tsne
    try:
        import umap
        print("UMAP ...")
        um = np.full((len(frames), 2), np.nan)
        um[sub] = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(P_keep[sub])
        embeddings["umap"] = um
    except ImportError:
        print("UMAP: umap-learn não instalado, pulado (pip install umap-learn)")

    gt_names = {"H": "casa", "A": "fora", "D": "bola morta"}
    sp_names = {1: "perto de reinício", 0: "jogo corrido (>30 s de reinício)", -1: "intermediário"}
    for method, emb in embeddings.items():
        idx = np.arange(len(frames)) if method == "pca" else sub
        e = emb[idx]
        scatter(e, cluster[idx], "{} ({}) — cluster GMM k={}".format(method, tag, k_best),
                os.path.join(OUT_DIR, "proj_{}_cluster.png".format(method)))
        if hdb_labels is not None:
            scatter(e, hdb_labels[idx], "{} — HDBSCAN (-1 = ruído)".format(method),
                    os.path.join(OUT_DIR, "proj_{}_hdbscan.png".format(method)), cmap="tab20")
        has_gt = gt_s[idx] != "X"
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.scatter(e[:, 0], e[:, 1], s=2, color="#dddddd", rasterized=True, label="sem GT")
        for lab, col in (("H", "#a50044"), ("A", "#1f77b4"), ("D", "#111111")):
            sel = has_gt & (gt_s[idx] == lab)
            ax.scatter(e[sel, 0], e[sel, 1], s=7, color=col, label="{} ({})".format(gt_names[lab], sel.sum()))
        ax.legend(markerscale=3, fontsize=8)
        ax.set_title("{} — GT público (5 min)".format(method))
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "proj_{}_gt.png".format(method)), dpi=130)
        plt.close(fig)
        scatter(e, sp_s[idx], "{} — Smart Tagging".format(method),
                os.path.join(OUT_DIR, "proj_{}_setpiece.png".format(method)),
                labels=sp_names, order=[0, -1, 1], cmap="Set1")
        scatter(e, ball_s[idx], "{} — bola visível no ATD".format(method),
                os.path.join(OUT_DIR, "proj_{}_bola.png".format(method)),
                labels={0.0: "sem bola", 1.0: "com bola"}, cmap="Set2")
        scatter(e, half_s[idx], "{} — tempo de jogo".format(method),
                os.path.join(OUT_DIR, "proj_{}_tempo.png".format(method)),
                labels={1.0: "1º tempo", 2.0: "2º tempo"}, cmap="Dark2")

    # ---- perfil dos clusters, aplicado a todos os frames avaliados ---------------------------
    frames_c = frames_all[complete_all]
    cluster_all = gmm.predict(pca.transform(scaler.transform(
        X[frames_c][:, [index[n] for n in names]]))[:, :n_keep])
    per_frame = np.full(n_frames, -1, dtype=np.int16)
    per_frame[frames_c] = cluster_all
    pd.DataFrame({"frame": frames_c, "cluster": cluster_all}).to_csv(
        os.path.join(OUT_DIR, "clusters_por_frame.csv"), index=False)

    durations = run_lengths_median(per_frame[graded])
    gt_all, sp_all = gt[frames_c], sp[frames_c]
    summary = []
    for c in range(k_best):
        sel = cluster_all == c
        g = gt_all[sel]
        n_gt = np.sum(g != "X")
        summary.append({
            "cluster": c,
            "pct_jogo": 100 * sel.mean(),
            "frames_gt": int(n_gt),
            "pct_D_no_gt": 100 * np.mean(g[g != "X"] == "D") if n_gt else np.nan,
            "pct_perto_reinicio": 100 * np.mean(sp_all[sel] == 1),
            "pct_jogo_corrido": 100 * np.mean(sp_all[sel] == 0),
            "duracao_mediana_s": durations.get(c, np.nan),
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(OUT_DIR, "clusters_resumo.csv"), index=False, float_format="%.1f")
    profile = pd.DataFrame(Z, columns=names).groupby(cluster).mean().T
    profile.to_csv(os.path.join(OUT_DIR, "clusters_perfil.csv"), float_format="%.2f")
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * k_best, 0.28 * len(names) + 1.5))
    im = ax.imshow(profile.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xticks(range(k_best))
    ax.set_xticklabels(["c{}\nD {:.0f}%".format(c, summary.loc[c, "pct_D_no_gt"])
                        if not np.isnan(summary.loc[c, "pct_D_no_gt"]) else "c{}".format(c)
                        for c in range(k_best)], fontsize=8)
    fig.colorbar(im, ax=ax, label="z-score médio")
    ax.set_title("perfil dos clusters ({})".format(tag))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "clusters_perfil.png"), dpi=130)
    plt.close(fig)

    # ---- separabilidade viva × morta ----------------------------------------------------------
    all_names = PLAYER_FEATURES + BALL_FEATURES + ["ball_present", "ball_gap_since_s", "ball_gap_len_s",
                                                   "ball_last_out_dist_m", "ball_still_s"]
    y_gt = (gt == "D").astype(int)
    rows_gt = auc_table(X, index, all_names, y_gt, graded & (gt != "X"))
    rows_sp = auc_table(X, index, all_names, (sp == 1).astype(int), graded & (sp >= 0))
    sep = pd.DataFrame({
        "feature": [r[0] for r in rows_gt],
        "auc_gt": [r[1] for r in rows_gt], "dir_gt": [r[2] for r in rows_gt],
        "mediana_D_gt": [r[3] for r in rows_gt], "mediana_viva_gt": [r[4] for r in rows_gt],
        "auc_setpiece": [r[1] for r in rows_sp], "dir_setpiece": [r[2] for r in rows_sp],
        "mediana_reinicio": [r[3] for r in rows_sp], "mediana_corrido": [r[4] for r in rows_sp],
        "cobertura": [r[5] for r in rows_gt],
    }).sort_values("auc_gt", ascending=False)
    sep.to_csv(os.path.join(OUT_DIR, "separabilidade.csv"), index=False, float_format="%.3f")
    top = sep.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    yy = np.arange(len(top))
    ax.barh(yy - 0.2, top["auc_gt"], height=0.4, label="GT público (D × viva)")
    ax.barh(yy + 0.2, top["auc_setpiece"], height=0.4, label="Smart Tagging (reinício × corrido)")
    ax.set_yticks(yy)
    ax.set_yticklabels(["{} ({})".format(f, d) for f, d in zip(top["feature"], top["dir_gt"])], fontsize=8)
    ax.axvline(0.5, color="grey", lw=0.8)
    ax.set_xlim(0.5, 1)
    ax.set_xlabel("AUC univariada (0,5 = não separa; '<' = menor em bola morta)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "separabilidade.png"), dpi=130)
    plt.close(fig)

    # ---- relatório ----------------------------------------------------------------------------
    with open(os.path.join(OUT_DIR, "relatorio.md"), "w", encoding="utf-8") as handle:
        handle.write("# Exploração de estados ({})\n\n".format(tag))
        handle.write("- {} instâncias, {} features\n".format(len(frames), len(names)))
        handle.write("- PCA: PC1 {:.1%}, PC2 {:.1%}; {} componentes para {:.0%}\n".format(
            pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1], n_keep, PCA_VARIANCE))
        handle.write("- GMM: k escolhido = {} (cotovelo do BIC){}\n".format(
            k_best, "" if n_hdb is None else "; HDBSCAN: {} clusters".format(n_hdb)))
        handle.write("\n## Clusters\n\n" + summary.to_markdown(index=False, floatfmt=".1f") + "\n")
        handle.write("\n## Cargas PC1/PC2 (top 8 em módulo)\n\n")
        for pc in ("PC1", "PC2"):
            s = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index).head(8)
            handle.write("- **{}**: ".format(pc) + ", ".join("{} {:+.2f}".format(k, v) for k, v in s.items()) + "\n")
        handle.write("\n## Separabilidade (top 20 por AUC no GT público)\n\n")
        handle.write(sep.head(20).to_markdown(index=False, floatfmt=".3f") + "\n")
    print("\n" + summary.to_string(index=False, float_format=lambda v: "{:.1f}".format(v)))
    print("\nsaída em", OUT_DIR)


if __name__ == "__main__":
    main()
