"""Testa outras configurações da TCN e da LSTM (largura, profundidade, dropout, alcance), com sementes.

    python -m posse_ml.testar_configs                                    # todas, 5 épocas, sementes 10 11 12
    python -m posse_ml.testar_configs --epocas 3                         # outro número de épocas
    python -m posse_ml.testar_configs --configs tcn_d03 lstm_c1 --epocas 3 5
    python -m posse_ml.testar_configs --validacao 5                      # validação sempre com 5 épocas
    python -m posse_ml.testar_configs --so-comparar                      # não treina; só refaz a tabela

As configurações (o que muda em relação à padrão; o resto fica igual):

  tcn_d03   TCN com dropout 0,3 (padrão 0,1): regulariza, pode proteger da especialização na DFL
  tcn_l32   TCN com largura 32 (padrão 64): rede menor
  tcn_l128  TCN com largura 128: rede maior
  tcn_b4    TCN com 4 blocos: alcance de ~±2,5 s (padrão 6 blocos, ~±10 s)
  tcn_b7    TCN com 7 blocos e janela de 512 passos: alcance de ~±20 s
  lstm_d03  LSTM com dropout 0,3
  lstm_l32  LSTM com largura 32
  lstm_l128 LSTM com largura 128
  lstm_c1   LSTM com 1 camada (padrão 2)
  lstm_c3   LSTM com 3 camadas

O número de épocas é livre (--epocas, um ou vários), e a validação usa as mesmas épocas do modelo
final (--validacao igual, padrão) ou sempre 5. Os modelos ficam em modelos/<instância>/<config>_f<N>
[v<N>]_s<semente>, com os mesmos nomes do testar_epocas. A configuração padrão com as mesmas épocas,
validação e sementes (do testar_epocas) entra como referência, se existir.

Para cada configuração e número de épocas: a média das sementes e o valor de cada uma, na DFL, na
janela (própria, rígido, suave), no teto e nas medidas por frame. A tabela sai na tela e em
modelos/<instância>/configs.md. Escolha pela mesma regra de sempre: média do spellF1 no rígido;
menos de 0,03 de diferença é empate, desempatado pelo macroF1; as medidas por frame conferem.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

from . import comparar_classificadores as C, config

CONFIGS = {
    "tcn_d03": ("tcn", "dropout=0.3", "dropout 0,3"),
    "tcn_l32": ("tcn", "largura=32", "largura 32"),
    "tcn_l128": ("tcn", "largura=128", "largura 128"),
    "tcn_b4": ("tcn", "blocos=4", "4 blocos (~±2,5 s)"),
    "tcn_b7": ("tcn", "blocos=7,janela=512", "7 blocos, janela 512 (~±20 s)"),
    "lstm_d03": ("lstm", "dropout=0.3", "dropout 0,3"),
    "lstm_l32": ("lstm", "largura=32", "largura 32"),
    "lstm_l128": ("lstm", "largura=128", "largura 128"),
    "lstm_c1": ("lstm", "camadas=1", "1 camada"),
    "lstm_c3": ("lstm", "camadas=3", "3 camadas"),
}


def nome(base, n, s, igual):
    """Como no testar_epocas: <base>_f<N>[v<N>][_s<semente>]; a TCN/LSTM padrão de 10 épocas validada
    com 5 é só <rede>[_s<semente>]."""
    if base in ("tcn", "lstm") and n == 10 and not igual:
        nm = base
    else:
        nm = "{}_f{}".format(base, n) + ("v{}".format(n) if igual and n != 5 else "")
    return nm if s == 0 else "{}_s{}".format(nm, s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--epocas", nargs="+", type=int, default=[5], help="épocas do modelo final (um ou vários)")
    ap.add_argument("--validacao", choices=("igual", "5"), default="igual",
                    help="épocas da validação que escolhe os custos: iguais às do modelo final, ou sempre 5")
    ap.add_argument("--sementes", nargs="+", type=int, default=[10, 11, 12])
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst, igual = args.instancia, args.validacao == "igual"
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para comparar".format(config.GT_PUBLICO))
    if importlib.util.find_spec("torch") is None:
        sys.exit("as redes precisam do PyTorch (requirements-classificadores.txt)")
    combos = [(cfg, n, s) for cfg in args.configs for n in args.epocas for s in args.sementes]

    # 1. treino do que falta
    falhas, tempos = [], {}
    for i, (cfg, n, s) in enumerate(combos, 1):
        clf, extra, _ = CONFIGS[cfg]
        nm = nome(cfg, n, s, igual)
        pasta = config.pasta_modelos(inst, nm)
        if C.treinado(pasta) or args.so_comparar:
            print("\n=== [{}/{}] {}: {}".format(i, len(combos), nm,
                                              "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
            continue
        if args.teste:
            ep = max(1, n // 5)
            hiper = "{},semente={},epocas={},epocas_rapido={}".format(extra, s, ep, ep if igual else 1)
        else:
            hiper = "{},semente={},epocas={},epocas_rapido={}".format(extra, s, n, n if igual else 5)
        print("\n=== [{}/{}] {}".format(i, len(combos), nm))
        ok, tempos[nm] = C.treinar_pasta(inst, clf, pasta, hiper, sem_gpu=args.sem_gpu,
                                         passo=3 if args.teste else None, nome=nm)
        if not ok:
            falhas.append(nm)

    # 2. janela pública: as configurações e a padrão de referência, com o jogo lido uma vez
    grupos = []                                  # (rótulo, descrição, épocas, [nomes por semente])
    for clf in ("tcn", "lstm"):
        if not any(CONFIGS[c][0] == clf for c in args.configs):
            continue
        for n in args.epocas:
            grupos.append((clf + " (referência)", "padrão", n, [nome(clf, n, s, igual) for s in args.sementes]))
        for cfg in args.configs:
            if CONFIGS[cfg][0] == clf:
                for n in args.epocas:
                    grupos.append((cfg, CONFIGS[cfg][2], n, [nome(cfg, n, s, igual) for s in args.sementes]))
    nomes_todos = [nm for g in grupos for nm in g[3] if C.treinado(config.pasta_modelos(inst, nm))]
    if not nomes_todos:
        sys.exit("nada treinado para comparar")
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    res = {}
    for nm in nomes_todos:
        try:
            res[nm] = C.avaliar(p, x, nomes, plano, gt, inst, nm, paradas, teto=True)
        except Exception as erro:            # um modelo com problema não derruba a comparação
            print("=== {}: falhou na comparação: {}".format(nm, erro))
            falhas.append(nm)

    # 3. tabela: média das sementes (e cada semente) por configuração e número de épocas
    def valor(ln, col):
        if col in ("frames_morta", "frames_ha"):
            return ln.get(col)
        if col == "dfl":
            return ln["validacao"][0]
        if col == "macro_rigido":
            return ln["rigido"][1] if ln.get("rigido") else None
        return ln[col][0] if ln.get(col) else None

    def celula(vals, detalhe=True):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        media = "**{}**".format(C._num(float(np.mean(vals))))
        return media + (" ({})".format(" / ".join(C._num(v) for v in vals)) if detalhe else "")

    cols = [("dfl", "DFL", False), ("rigido", "janela, rígido", True), ("macro_rigido", "macroF1, rígido", False),
            ("suave", "janela, suave", True), ("teto", "teto (GT, não vale)", False),
            ("propria", "janela, própria", False), ("frames_morta", "por frame: morta/viva", False),
            ("frames_ha", "por frame: H/A", False)]
    cab = ["configuração", "o que muda", "épocas", "sementes"] + [t for _, t, _ in cols]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for rotulo, desc, n, nms in grupos:
        lns = [res[nm] for nm in nms if nm in res]
        if not lns:
            continue
        tab.append("| {} | {} | {} | {} | {} |".format(rotulo, desc, n, len(lns), " | ".join(
            celula([valor(ln, k) for ln in lns], det) for k, _, det in cols)))
    nota = ("Cada célula: **média das sementes**; no rígido e no suave, também o valor de cada semente. Janela: "
            "spellF1. Validação {}. Regra: média do spellF1 no rígido; menos de 0,03 é empate, desempatado pelo "
            "macroF1; as medidas por frame conferem.").format(
                "com as mesmas épocas do modelo final" if igual else "sempre com 5 épocas")
    texto = "\n".join(tab) + "\n\n" + nota
    print("\n" + texto)
    destino = os.path.join(config.pasta_modelos(inst), "configs.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Configurações da TCN e da LSTM, instância {}\n\n{}\n".format(inst, texto))
    print("\ntabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
