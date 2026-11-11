"""Compara fontes de bola morta com o GT público: o detector do Nuno e as submissões do posse_ml.

    python -m posse_ml.comparar_paradas
    python -m posse_ml.comparar_paradas --arquivos outra.csv mais_uma.csv

Para cada fonte, na janela pública:

  acerto        fração dos frames com morta/viva certo
  precisão D    dos frames que a fonte chama de bola morta, quantos são
  recall D      dos frames de bola morta do GT, quantos a fonte pegou
  paradas       quantas paradas (trechos de D) a fonte tem na janela, e quantas o GT tem
  inventadas    paradas da fonte que não encostam em nenhuma parada do GT
  perdidas      paradas do GT que não encostam em nenhuma parada da fonte
  início / fim  para cada parada do GT, quanto a borda da fonte está atrasada (+) ou adiantada (-),
                em frames, quando há uma borda a até 5 s

A janela tem poucas paradas: serve para ver diferenças grandes e onde cada fonte erra, não para
ranquear fontes parecidas.
"""

import argparse
import glob
import os

import numpy as np

from . import config
from .config import FPS
from .janelas import trechos
from .prever import ler_paradas

JANELA_BORDA = 5 * FPS


def _bordas(morta):
    d = np.diff(np.r_[0, morta.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _atrasos(ref, prev):
    out, perdidas = [], 0
    for x in ref:
        if len(prev) and np.min(np.abs(prev - x)) <= JANELA_BORDA:
            out.append(int(prev[np.argmin(np.abs(prev - x))] - x))
        else:
            perdidas += 1
    return out, perdidas


def _encostam(a, b):
    """Quantos trechos de `a` encostam em algum trecho de `b`."""
    return sum(1 for s, e in a if any(s < e2 and s2 < e for s2, e2 in b))


def comparar(nome, forca, gt_morta, janela):
    a, b = np.flatnonzero(janela)[[0, -1]]
    sl = slice(a, b + 1)
    morta = forca[sl] == 1
    ref = gt_morta[sl]
    conhecida = forca[sl] >= 0
    tg, tp = trechos(ref), trechos(morta)
    ini_g, fim_g = _bordas(ref)
    ini_p, fim_p = _bordas(morta)
    atr_ini, _ = _atrasos(ini_g, ini_p)
    atr_fim, _ = _atrasos(fim_g, fim_p)
    return {
        "fonte": nome,
        "acerto": float(np.mean(morta[conhecida] == ref[conhecida])) if conhecida.any() else np.nan,
        "sem informação": float(np.mean(~conhecida)),
        "precisão D": float(ref[morta].mean()) if morta.any() else np.nan,
        "recall D": float(morta[ref].mean()) if ref.any() else np.nan,
        "paradas": "{} (GT {})".format(len(tp), len(tg)),
        "inventadas": len(tp) - _encostam(tp, tg),
        "perdidas": len(tg) - _encostam(tg, tp),
        "início": sorted(atr_ini),
        "fim": sorted(atr_fim),
    }


def main(argv=None):
    from .__main__ import pm
    from games import GAME, source_frames

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arquivos", nargs="*", help="outras fontes (frame,label ou frame,morta)")
    args = ap.parse_args(argv)

    n = source_frames("tactical", GAME)
    gt = pm.label_codes(pm.read_label_csv(config.GT_PUBLICO)).astype(np.int8)
    janela = gt != config.X
    gt_morta = gt == config.D

    fontes = []
    if os.path.isfile(config.PARADAS_EQUIPE):
        fontes.append(("nuno (paradas_D)", config.PARADAS_EQUIPE))
    for c in sorted(glob.glob(os.path.join(config.RESULTADOS, "posse_ml_*.csv"))):
        if "_gt_" not in os.path.basename(c):          # o teste do teto usa o próprio GT
            fontes.append((os.path.basename(c)[:-4], c))
    for c in args.arquivos or []:
        fontes.append((os.path.basename(c)[:-4], c))
    if not fontes:
        raise SystemExit("nenhuma fonte: nem {} nem evaluation/results/posse_ml_*.csv".format(config.PARADAS_EQUIPE))

    linhas = [comparar(nome, ler_paradas(c, n), gt_morta, janela) for nome, c in fontes]
    largura = max(len(l["fonte"]) for l in linhas) + 2
    print("\n{:<{w}}{:>8}{:>9}{:>9}{:>13}{:>12}{:>10}".format(
        "", "acerto", "prec. D", "recall D", "paradas", "inventadas", "perdidas", w=largura))
    for l in linhas:
        print("{:<{w}}{:>7.1%} {:>8.1%} {:>8.1%} {:>13}{:>12}{:>10}".format(
            l["fonte"], l["acerto"], l["precisão D"], l["recall D"], l["paradas"], l["inventadas"],
            l["perdidas"], w=largura))
    print("\nbordas (fonte - GT, em frames; + é atraso):")
    for l in linhas:
        print("  {:<{w}} início {}  fim {}".format(l["fonte"], l["início"], l["fim"], w=largura))
    sem = [l for l in linhas if l["sem informação"] > 0]
    for l in sem:
        print("  ({}: sem informação em {:.1%} dos frames da janela, fora da conta do acerto)".format(
            l["fonte"], l["sem informação"]))


if __name__ == "__main__":
    main()
