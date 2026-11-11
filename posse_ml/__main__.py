"""python -m posse_ml [--instancia viterbi|fusao|atd] [--classificador hgb|xgboost|...] [--out CSV]

Escreve a submissão com os modelos que já estão em posse_ml/modelos (treinar.py) e os dados em
./data (ou $MTAH_DATA_DIR). Depois de gravar, confere o arquivo com o validate_submission do
avaliador oficial e, se o GT público estiver no repositório, mostra o placar nos 5 minutos dele.
"""

import argparse
import os
import sys
import time

import numpy as np

from . import config, prever, transmissao
from .partida import carregar_jogo

sys.path.insert(0, os.path.join(config.RAIZ, "evaluation"))
import possession_metrics as pm  # noqa: E402


def placar_publico(rot):
    """(spellF1, macroF1, posses previstas, posses no GT) na janela pública, ou None."""
    if not os.path.isfile(config.GT_PUBLICO):
        return None
    gt = pm.label_codes(pm.read_label_csv(config.GT_PUBLICO))
    pred = np.where(gt == config.X, config.X, rot).astype(np.int8)
    r = pm.evaluate(gt.astype(np.int8), pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m posse_ml", description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    config.argumento_classificador(ap)
    ap.add_argument("--bola", help="outro CSV da bola; padrão: a bola da instância (a mesma do medir_bola)")
    ap.add_argument("--out", help="padrão: evaluation/results/posse_ml_<instância>.csv")
    ap.add_argument("--sem-transmissao", action="store_true", help="não usa o vídeo de transmissão")
    ap.add_argument("--sem-goleiros", action="store_true", help="não corrige o time dos goleiros")
    ap.add_argument("--modelos", help="padrão: modelos/<instância>")
    ap.add_argument("--paradas", help="CSV de bola morta de fora: frame,morta (analise/out/paradas_D.csv) "
                    "ou frame,label (o GT público, para o teste do teto)")
    ap.add_argument("--modo-paradas", choices=("rigido", "suave"), default="rigido",
                    help="rigido: a bola morta é a do CSV; suave: o CSV é mais uma evidência "
                         "(precisa do calibrar com o mesmo --paradas)")
    args = ap.parse_args(argv)
    args.bola = args.bola or config.bola_da_instancia(args.instancia)
    args.modelos = args.modelos or config.pasta_modelos(args.instancia, args.classificador)
    args.out = args.out or config.nome_saida(args.instancia, args.classificador, args.paradas,
                                             args.modo_paradas if args.paradas else None)

    t0 = time.time()
    p = carregar_jogo(bola=args.bola, goleiros=not args.sem_goleiros)
    probs = prever.probabilidades(p, args.modelos)
    plano = None
    if not args.sem_transmissao:
        plano, desl = transmissao.do_jogo(p)
        print("transmissão: tático = transmissão + {}".format(" / ".join(str(d) for d in desl)))
    rot = prever.rotular(p, probs, plano, args.modelos, paradas=args.paradas,
                         modo=args.modo_paradas if args.paradas else None)
    prever.gravar(args.out, rot)

    problema = pm.validate_submission(pm.read_label_csv(args.out), len(rot))
    if problema:
        sys.exit("gravado, mas INVÁLIDO: {}".format(problema))
    conta = {letra: int(np.sum(rot == i)) for i, letra in enumerate(config.LETRAS)}
    print("gravado {}  H {H}  A {A}  D {D}  X {X}  ({:.0f} s)".format(args.out, time.time() - t0, **conta))
    pl = placar_publico(rot)
    if pl:
        print("janela pública: spellF1 {:.3f}  macroF1 {:.3f}  posses {:.0f} (GT {:.0f})".format(*pl))


if __name__ == "__main__":
    main()
