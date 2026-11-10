"""Compara as bolas (ATD, fusão, Viterbi) na janela pública e no jogo inteiro.

    python -m posse_ml.comparar_bolas

Não há posição verdadeira da bola nesses frames, então a comparação usa o que existe:

  presença       fração dos frames com bola, com o jogo rolando e parado
  pulos          fração das posições a mais de 3 m da mediana de ±5 frames
  lacunas >= 2 s quantas lacunas de 2 s ou mais com o jogo rolando
  mais perto     com o jogo rolando e bola presente, em que fração dos frames o jogador mais perto
                 da bola é do time que tem a posse no GT; uma bola mais certa deve ficar mais
                 vezes junto do time que a tem (só na janela pública, que tem GT de posse)
  ao dono        mediana da distância da bola ao jogador mais perto do time com a posse

Janela pública: bola viva/morta e posse pelo GT. Jogo inteiro: viva/morta pelo
analise/out/paradas_D.csv (se existir). No fim, a distância mediana entre as bolas onde as duas
existem, na janela.
"""

import os
import warnings

import numpy as np

from . import config
from .bola import remover_pulos
from .config import A, D, FPS, H, MANDANTE, VISITANTE, X
from .janelas import trechos
from .medir_bola import ler_rotulos
from .partida import carregar_jogo, ler_bola_csv


def _bolas(p):
    """{nome: bola em metros, sem os pulos}, só as que existem no disco."""
    escala = np.array(p.campo, dtype=np.float32)
    out = {"atd": p.bola}
    for nome, bola in (("fusao", "fusao"), ("viterbi", None)):
        caminho = config.caminho_bola(bola)
        if os.path.isfile(caminho):
            out[nome] = ler_bola_csv(caminho, p.n) * escala
        else:
            print("({} não encontrada em {})".format(nome, caminho))
    return out


def _lacunas_longas(vista, mascara_viva, tempos, minimo=2 * FPS):
    n = 0
    for _, a, b in tempos:
        for s, e in trechos(~vista[a:b + 1]):
            if e - s >= minimo and mascara_viva[a + s:a + e].mean() > 0.5:
                n += 1
    return n


def _perto_do_dono(p, bola, gt, quadros):
    """(fração em que o mais perto é do time com a posse, mediana da distância ao dono)."""
    q = quadros[np.isfinite(bola[quadros, 0])]
    d = np.hypot(p.jogadores[q, :, 0] - bola[q, None, 0], p.jogadores[q, :, 1] - bola[q, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    lado = p.lado[q]
    dono = np.where(gt[q] == H, MANDANTE, VISITANTE)
    mais_perto = lado[np.arange(len(q)), np.argmin(d, axis=1)]
    ao_dono = np.where(lado == dono[:, None], d, np.inf).min(axis=1)
    ao_dono = ao_dono[np.isfinite(ao_dono)]
    return float(np.mean(mais_perto == dono)), float(np.median(ao_dono)) if len(ao_dono) else np.nan


def main(argv=None):
    from .__main__ import pm as metricas   # o avaliador oficial (evaluation/possession_metrics.py)

    p = carregar_jogo(bola="atd")
    bolas = _bolas(p)
    gt = metricas.label_codes(metricas.read_label_csv(config.GT_PUBLICO)).astype(np.int8)
    janela = gt != X
    estados = ler_rotulos(config.PARADAS_EQUIPE, p.n) if os.path.isfile(config.PARADAS_EQUIPE) else None
    avaliado = p.jogados()

    linhas = []
    for nome, b in bolas.items():
        limpa, pulo = remover_pulos(b)
        vista = np.isfinite(limpa[:, 0])
        presente = np.isfinite(b[:, 0])
        viva_j = janela & ((gt == H) | (gt == A))
        morta_j = janela & (gt == D)
        perto, dono = _perto_do_dono(p, limpa, gt, np.flatnonzero(viva_j))
        lin = {"bola": nome,
               "janela: viva": vista[viva_j].mean(), "janela: morta": vista[morta_j].mean(),
               "janela: lacunas >= 2 s viva": _lacunas_longas(vista, viva_j, p.tempos),
               "janela: mais perto é o dono": perto, "janela: ao dono (m)": dono,
               "pulos": pulo[presente & avaliado].mean() if (presente & avaliado).any() else 0.0,
               "jogo: presença": vista[avaliado].mean()}
        if estados is not None:
            viva = avaliado & (estados != D) & (estados != X)
            morta = avaliado & (estados == D)
            lin.update({"jogo: viva": vista[viva].mean(), "jogo: morta": vista[morta].mean(),
                        "jogo: lacunas >= 2 s viva": _lacunas_longas(vista, viva, p.tempos)})
        lin["_limpa"] = limpa
        linhas.append(lin)

    chaves = [k for k in linhas[0] if not k.startswith("_") and k != "bola"]
    print("\n{:<30}".format("") + "".join("{:>10}".format(l["bola"]) for l in linhas))
    for k in chaves:
        vals = []
        for l in linhas:
            v = l.get(k, np.nan)
            if isinstance(v, (int, np.integer)):
                vals.append("{:>10d}".format(int(v)))
            elif "(m)" in k:
                vals.append("{:>10.2f}".format(v))
            else:
                vals.append("{:>9.1%} ".format(v))
        print("{:<30}".format(k) + "".join(vals))
    print("(viva/morta e presença: fração dos frames com bola, depois do filtro de pulos)")

    print("\ndistância mediana entre as bolas na janela, onde as duas existem (fração a menos de 2 m):")
    for i in range(len(linhas)):
        for j in range(i + 1, len(linhas)):
            a, b = linhas[i]["_limpa"], linhas[j]["_limpa"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                d = np.hypot(*(a[janela] - b[janela]).T)
            d = d[np.isfinite(d)]
            if len(d):
                print("  {} x {}: {:.2f} m ({:.0%}), {} frames".format(
                    linhas[i]["bola"], linhas[j]["bola"], np.median(d), np.mean(d < 2), len(d)))


if __name__ == "__main__":
    main()
