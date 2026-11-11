"""Treina variantes das redes e compara com as que já existem.

    python -m posse_ml.testar_variantes                       # todas (treina só as que faltam)
    python -m posse_ml.testar_variantes --variantes tcn_e5 lstm_e5 lstm_e20
    python -m posse_ml.testar_variantes --so-comparar         # não treina; só refaz a tabela

As variantes (cada uma em modelos/<instância>/<nome>/):

  lstm         LSTM bidirecional com a mesma configuração da GRU (5 épocas na validação, 10 no final)
  lstm_e5      LSTM com 5 épocas, na validação e no modelo final
  lstm_e20     LSTM com 20 épocas, na validação e no modelo final
  tcn_e5       TCN com 5 épocas, na validação e no modelo final
  tcn_e10      a TCN de sempre, mas validada com as mesmas 10 épocas do modelo final (a tcn padrão é
               validada com 5)
  tcn_e20      TCN com 20 épocas, na validação e no modelo final
  tcn_e30      TCN com 30 épocas, na validação e no modelo final
  transformer  encoder pequeno (3 camadas, 4 cabeças, largura 64) com atenção na janela de ~20 s e
               viés de posição relativa; mesmas épocas da GRU e da TCN (5 na validação, 10 no final)

Fora as épocas, nada muda: mesma largura, profundidade, janela e taxa de aprendizado. A tcn e a gru
que já existem entram na tabela como referência, sem treinar de novo.

Para cada uma: validação na DFL, as três versões na janela pública (submissões gravadas como
posse_ml_<instância>_<nome>_...), o teto (bola morta do GT, sem gravar arquivo) e duas medidas por
frame na janela, tiradas do modelo antes do Viterbi: o acerto de bola morta/viva e o acerto de
H/A com a bola viva. Elas usam ~7.500 frames em vez de 17 posses e oscilam bem menos que o
spellF1, mas frames vizinhos não são independentes. A tabela sai na tela e em
modelos/<instância>/variantes.md.

Cuidado com as épocas: na DFL, treinar mais sempre parece melhor, porque a validação também é DFL;
no nosso jogo, 20 e 30 épocas pioraram (README, seção 5.10). Para épocas, a DFL não decide sozinha.
"""

import argparse
import importlib.util
import os
import sys

from . import comparar_classificadores as C, config

VARIANTES = {
    "lstm": ("lstm", ""),
    "tcn_e10": ("tcn", "epocas=10"),
    "tcn_e20": ("tcn", "epocas=20"),
    "tcn_e30": ("tcn", "epocas=30"),
    "transformer": ("transformer", ""),
    "tcn_e5": ("tcn", "epocas=5"),
    "lstm_e5": ("lstm", "epocas=5"),
    "lstm_e20": ("lstm", "epocas=20"),
}
REFERENCIAS = ("tcn", "gru")


def _pequeno(hiper):
    """--teste: as mesmas variantes com um décimo das épocas (no mínimo 1; 2 quando não diz)."""
    d = dict(item.split("=") for item in hiper.split(",") if item)
    d["epocas"] = str(max(1, int(d["epocas"]) // 10)) if "epocas" in d else "2"
    return ",".join("{}={}".format(k, v) for k, v in d.items())


def _epocas(hiper):
    for item in hiper.split(","):
        if item.startswith("epocas="):
            n = item.split("=")[1]
            return "{0} / {0}".format(n)
    return "5 / 10"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--variantes", nargs="+", choices=list(VARIANTES), default=list(VARIANTES))
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst = args.instancia
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para calibrar e comparar".format(config.GT_PUBLICO))
    if importlib.util.find_spec("torch") is None:
        sys.exit("as variantes são redes e precisam do PyTorch (requirements-classificadores.txt)")

    # 1. treino das variantes que faltam
    falhas, tempos = [], {}
    for nome in args.variantes:
        clf, hiper = VARIANTES[nome]
        pasta = config.pasta_modelos(inst, nome)
        if C.treinado(pasta) or args.so_comparar:
            print("\n=== {}: {}".format(nome, "já treinado" if C.treinado(pasta) else "não treinado, pulando"))
            continue
        ok, tempos[nome] = C.treinar_pasta(inst, clf, pasta, _pequeno(hiper) if args.teste else hiper,
                                           sem_gpu=args.sem_gpu, passo=3 if args.teste else None, nome=nome)
        if not ok:
            falhas.append(nome)

    # 2. janela pública: referências e variantes, com o jogo lido uma vez
    versoes = [r for r in REFERENCIAS if C.treinado(config.pasta_modelos(inst, r))]
    versoes += [v for v in args.variantes if C.treinado(config.pasta_modelos(inst, v))]
    if not versoes:
        sys.exit("nada treinado para comparar")
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    linhas = [C.avaliar(p, x, nomes, plano, gt, inst, v, paradas, teto=True) for v in versoes]

    cab = ["modelo", "épocas (validação / final)", "treino", "validação DFL (spellF1 / macroF1)",
           "janela, bola morta própria", "janela, detector rígido", "janela, detector suave",
           "teto (bola morta do GT, não vale)", "por frame: morta/viva", "por frame: H/A (bola viva)"]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for ln in linhas:
        nome = ln["clf"]
        hiper = VARIANTES[nome][1] if nome in VARIANTES else ""
        v = ln["validacao"]
        tab.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            nome + ("" if nome in VARIANTES else " (referência)"), _epocas(hiper), tempos.get(nome, "já treinado"),
            "{} / {}".format(C._num(v[0]), C._num(v[1])) if v[0] else "—",
            C._placar(ln.get("propria")), C._placar(ln.get("rigido")), C._placar(ln.get("suave")),
            C._placar(ln.get("teto")), C._num(ln["frames_morta"]), C._num(ln["frames_ha"])))
    texto = "\n".join(tab)
    print("\n" + texto + "\n\n" + C.NOTA)
    destino = os.path.join(config.pasta_modelos(inst), "variantes.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Variantes das redes, instância {}\n\n{}\n\n{}\n".format(inst, texto, C.NOTA))
    print("tabela gravada em", destino)
    if falhas:
        sys.exit("com problema: " + ", ".join(falhas))


if __name__ == "__main__":
    main()
