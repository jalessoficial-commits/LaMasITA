"""Roda o preditor com cada bola (fusão, Viterbi, Kalman), com e sem filtro de pulos, e compara.

    python posse/comparar_bolas.py
    python posse/comparar_bolas.py --extra --touch-weight 1.0      # repassa opções ao preditor

Tudo nos 5 minutos públicos, com as fronteiras de bola morta do GT (--live-ha), e com as demais
melhorias do preditor ligadas (goleiro, portão). O ATD é lido uma vez só. Bolas cujo arquivo não
existe são puladas com um aviso.

Cada rodada grava a submissão em posse/out/comparacao_<bola>_<filtro>.csv (sobrescrita a cada
execução) e a camada posse/out/toques.csv fica com a ÚLTIMA rodada. No fim, uma tabela:

  spellF1, macroF1      do avaliador oficial
  posses                previstas (o GT tem 17)
  certas / perdidas     trocas do GT acertadas (até 2 s, mesmo sentido) e perdidas (de 12)
  inventadas            trocas previstas sem par no GT
  bloqueadas            trocas do GT que o portão não liberava

Com 12 trocas, diferenças de poucos pontos no spellF1 são ruído (uma posse a mais ou a menos move
~6 pontos): olhe também certas, perdidas e inventadas.
"""

import argparse
import contextlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import predictor as pr  # noqa: E402

GT = os.path.join("evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BALLS = (  # (nome, caminho a partir da raiz do repositório, unidades)
    ("fusao", os.path.join("data", "tactical", "BAR-ATH_tactical_ball_tracker.csv"), "norm"),
    ("viterbi", os.path.join("tracking_bola", "out", "viterbi_bola.csv"), "px"),
    ("kalman", os.path.join("tracking_bola", "out", "kalman_bola.csv"), "px"),
)
ROW = re.compile(r"^\s+(\d+)\s+([HA])\s+(.+?)\s+(sim|NÃO)?\s*$")


def run(argv):
    """Roda predictor.main() com estes argumentos; devolve o texto que ele imprimiu."""
    buffer = io.StringIO()
    old = sys.argv
    sys.argv = ["predictor.py"] + argv
    try:
        with contextlib.redirect_stdout(buffer):
            pr.main()
    finally:
        sys.argv = old
    return buffer.getvalue()


def parse(text):
    out = {}
    m = re.search(r"spellF1 ([\d.]+)\s+macroF1 ([\d.]+)\s+posses previstas (\d+)", text)
    if m:
        out["spell"], out["macro"], out["posses"] = float(m.group(1)), float(m.group(2)), int(m.group(3))
    rows = [ROW.match(line) for line in text.splitlines()]
    rows = [r for r in rows if r]
    out["certas"] = sum("frames" in r.group(3) for r in rows)
    out["perdidas"] = sum("frames" not in r.group(3) for r in rows)
    out["bloqueadas"] = sum(r.group(4) == "NÃO" for r in rows)
    m = re.search(r"previstas sem par no GT \(frames\): (.+)", text)
    out["inventadas"] = len(m.group(1).split(",")) if m else 0
    m = re.search(r"posições da bola descartadas por pular: \d+ de \d+ \(([\d.]+)%\)", text)
    out["descartadas"] = m.group(1) + "%" if m else "-"
    m = re.search(r"^goleiros: (.+)$", text, re.M)
    out["goleiros"] = m.group(1) if m else ""
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="opções repassadas ao preditor em todas as rodadas (vai por último)")
    args = parser.parse_args()
    os.chdir(ROOT)

    results = []
    for name, path, units in BALLS:
        if not os.path.isfile(path):
            print("pulando {}: não achei {}".format(name, path))
            continue
        for despike in (True, False):
            label = "{} {}".format(name, "com filtro" if despike else "sem filtro")
            print("rodando: {} ...".format(label), flush=True)
            argv = ["--dead", GT, "--live-ha", "--ball", path, "--ball-units", units,
                    "--out", os.path.join(HERE, "out", "comparacao_{}_{}.csv".format(
                        name, "filtro" if despike else "semfiltro"))]
            if not despike:
                argv.append("--no-despike")
            info = parse(run(argv + args.extra))
            info["nome"] = label
            results.append(info)

    if not results:
        print("nenhuma bola encontrada.")
        return
    if results[0].get("goleiros"):
        print("\ngoleiros: " + results[0]["goleiros"])
    print("\n{:<22} {:>8} {:>8} {:>7} {:>7} {:>9} {:>11} {:>11} {:>12}".format(
        "bola", "spellF1", "macroF1", "posses", "certas", "perdidas", "inventadas", "bloqueadas",
        "descartadas"))
    for r in sorted(results, key=lambda r: -r.get("spell", 0)):
        print("{:<22} {:>8.4f} {:>8.4f} {:>7} {:>7} {:>9} {:>11} {:>11} {:>12}".format(
            r["nome"], r.get("spell", float("nan")), r.get("macro", float("nan")),
            r.get("posses", "-"), r["certas"], r["perdidas"], r["inventadas"], r["bloqueadas"],
            r["descartadas"]))
    print("\nGT público: 17 posses, 12 trocas com a bola viva. Fronteiras de bola morta do próprio GT.")


if __name__ == "__main__":
    main()