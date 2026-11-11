"""Compara os dados de treino: só DFL, só Metrica e DFL + Metrica, medindo no GT público.

    python -m posse_ml.baixar_metrica                 # uma vez
    python -m posse_ml.metrica                        # confere os rótulos derivados contra o GT
    python -m posse_ml.testar_dados                   # tcn e lstm, 5 épocas, sementes 10 11 12
    python -m posse_ml.testar_dados --classificadores tcn --sementes 10 11 12 20 21 22
    python -m posse_ml.testar_dados --so-comparar
    python -m posse_ml.testar_dados --dados metrica --epocas 12       # só um treino, outro nº de épocas
    python -m posse_ml.testar_dados --dados dfl+metrica --epocas 3

Uma época é uma passada pelos dados: com os 3 jogos da Metrica, 5 épocas dão as atualizações de ~2
épocas na DFL; com os 10 jogos, as de ~7. Para igualar a quantidade de treino da DFL com 5 épocas:
~12 épocas para "metrica" e 3 ou 4 para "dfl+metrica". Com --dados, só os treinos pedidos são
feitos; a DFL com 5 épocas entra sempre na tabela como referência (se existir), e a tabela vai para
dados_f<N>.md quando N não é 5.

Para cada rede e semente, três modelos com a mesma configuração (N épocas na validação e no modelo
final), mudando só os jogos de treino:

  dfl          os 7 jogos da DFL: é o <rede>_f5_s<semente> do testar_epocas, reaproveitado
  metrica      os 3 jogos da Metrica                -> <rede>_f5_met_s<semente>
  dfl+metrica  os 10 jogos                          -> <rede>_f5_dflmet_s<semente>

Os jogos da Metrica passam pela mesma degradação da DFL (a bola deles é limpa com o jogo rolando e
ausente na bola parada; a degradação põe uma bola de parada e depois as falhas do perfil da nossa
bola). Os custos do Viterbi de cada modelo saem, como sempre, da validação que deixa um jogo de
treino de fora por vez; por isso a coluna "validação" não se compara entre linhas (os jogos são
outros). Quem compara os três treinos é a janela pública: média das sementes e o valor de cada uma,
no rígido, no suave, no teto e nas medidas por frame. Tabela em modelos/<instância>/dados.md.
"""

import argparse
import importlib.util
import os
import sys

import numpy as np

from . import comparar_classificadores as C, config, metrica

DADOS = (("dfl", ""), ("metrica", "_met"), ("dfl+metrica", "_dflmet"))


def nome(clf, n, dados, s):
    return "{}_f{}{}_s{}".format(clf, n, dict(DADOS)[dados], s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificadores", nargs="+", choices=("tcn", "lstm", "gru", "transformer"), default=["tcn", "lstm"])
    ap.add_argument("--epocas", type=int, default=5, help="épocas, na validação e no modelo final")
    ap.add_argument("--dados", nargs="+", choices=[d for d, _ in DADOS], default=[d for d, _ in DADOS],
                    help="quais treinos fazer (padrão: os três)")
    ap.add_argument("--sementes", nargs="+", type=int, default=[10, 11, 12])
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE)
    args = ap.parse_args(argv)
    inst, n = args.instancia, args.epocas
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para comparar".format(config.GT_PUBLICO))
    if importlib.util.find_spec("torch") is None:
        sys.exit("as redes precisam do PyTorch (requirements-classificadores.txt)")
    if not args.so_comparar and len(metrica.jogos()) < 2:
        sys.exit("não achei os jogos da Metrica em {} (python -m posse_ml.baixar_metrica)".format(config.METRICA))
    combos = [(c, d, s) for c in args.classificadores for d in args.dados for s in args.sementes]

    falhas, tempos = [], {}
    for i, (clf, dados, s) in enumerate(combos, 1):
        nm = nome(clf, n, dados, s)
        pasta = config.pasta_modelos(inst, nm)
        if C.treinado(pasta) or args.so_comparar:
            print("\n=== [{}/{}] {}: {}".format(i, len(combos), nm,
                                              "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
            continue
        ep = max(1, n // 5) if args.teste else n
        hiper = "semente={},epocas={},epocas_rapido={}".format(s, ep, ep)
        print("\n=== [{}/{}] {} (treino: {})".format(i, len(combos), nm, dados))
        ok, tempos[nm] = C.treinar_pasta(inst, clf, pasta, hiper, sem_gpu=args.sem_gpu,
                                         passo=3 if args.teste else None, nome=nm, extra=["--dados", dados])
        if not ok:
            falhas.append(nm)

    prontos = [c for c in combos if C.treinado(config.pasta_modelos(inst, nome(c[0], n, c[1], c[2])))]
    if not prontos:
        sys.exit("nada treinado para comparar")
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    res, ref = {}, {}
    for clf, dados, s in prontos:
        nm = nome(clf, n, dados, s)
        try:
            res[(clf, dados, s)] = C.avaliar(p, x, nomes, plano, gt, inst, nm, paradas, teto=True)
        except Exception as erro:
            print("=== {}: falhou na comparação: {}".format(nm, erro))
            falhas.append(nm)
    if n != 5 or "dfl" not in args.dados:          # a DFL com 5 épocas, como referência
        for clf in args.classificadores:
            for s in args.sementes:
                nm = nome(clf, 5, "dfl", s)
                if C.treinado(config.pasta_modelos(inst, nm)):
                    ref[(clf, s)] = C.avaliar(p, x, nomes, plano, gt, inst, nm, paradas, teto=True)

    def valor(ln, col):
        if col in ("frames_morta", "frames_ha"):
            return ln.get(col)
        if col == "val":
            return ln["validacao"][0]
        if col == "macro_rigido":
            return ln["rigido"][1] if ln.get("rigido") else None
        if col == "posses":
            return ln["rigido"][2] if ln.get("rigido") else None
        return ln[col][0] if ln.get(col) else None

    def celula(vals, detalhe, casas=3):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "—"
        f = C._num if casas == 3 else (lambda v: "{:.1f}".format(v).replace(".", ","))
        return "**{}**".format(f(float(np.mean(vals)))) + (
            " ({})".format(" / ".join(f(v) for v in vals)) if detalhe else "")

    cols = [("val", "validação (jogos de treino)", False), ("rigido", "janela, rígido", True),
            ("macro_rigido", "macroF1, rígido", False), ("posses", "posses no rígido (GT 17)", False),
            ("suave", "janela, suave", True), ("teto", "teto (GT, não vale)", False),
            ("propria", "janela, própria", False), ("frames_morta", "por frame: morta/viva", False),
            ("frames_ha", "por frame: H/A", False)]
    cab = ["rede", "treino", "épocas", "sementes"] + [t for _, t, _ in cols]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]

    def linha(clf, dados, ep, lns):
        tab.append("| {} | {} | {} | {} | {} |".format(clf, dados, ep, len(lns), " | ".join(
            celula([valor(ln, k) for ln in lns], det, 1 if k == "posses" else 3) for k, _, det in cols)))

    for clf in args.classificadores:
        lns = [ref[(clf, s)] for s in args.sementes if (clf, s) in ref]
        if lns:
            linha(clf, "dfl (referência)", 5, lns)
        for dados in args.dados:
            lns = [res[(clf, dados, s)] for s in args.sementes if (clf, dados, s) in res]
            if lns:
                linha(clf, dados, n, lns)
    nota = ("Cada célula: **média das sementes**; no rígido e no suave, também o valor de cada semente. Janela: "
            "spellF1. A coluna de validação usa os jogos de treino de cada linha e não se compara entre linhas. "
            "Uma semente oscila ~0,02 a 0,04 na janela: menos de 0,03 de diferença é empate.")
    texto = "\n".join(tab) + "\n\n" + nota
    print("\n" + texto)
    destino = os.path.join(config.pasta_modelos(inst), "dados.md" if n == 5 else "dados_f{}.md".format(n))
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Dados de treino ({} épocas), instância {}\n\n{}\n".format(n, inst, texto))
    print("\ntabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
