"""Tabela de métricas para comparar trackers de bola (ATD, YOLO, WASB, FootAndBall...) contra as
suas anotações à mão.

Colocar em tracking_bola/, junto de eval_atd.py e check_projection.py (este script importa deles).
Só numpy:

    python tracking_bola/compare_trackers.py
    python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo.csv \
                                             --tracker wasb=tracking_bola/out/wasb.csv --tol 25

O ATD já vem embutido (bola do ATD projetada em pixels com a homografia). Cada outro tracker é um
CSV com UMA bola por frame, em pixels do vídeo ORIGINAL (se o detector rodou em recortes, converta
de volta antes), no mesmo índice de frame da annotate_ball.py (começa em 0):

    frame,x,y            (coluna opcional: conf)

Frame sem linha no CSV = o tracker disse "sem bola" naquele frame. Por isso rode o tracker em TODOS
os frames anotados, inclusive onde ele não acha nada. Se houver várias linhas no mesmo frame, vale a
de maior conf (ou a primeira, se não houver conf).

Como cada frame com bola visível é classificado (tolerância --tol, padrão 25 px):
  Acerto        o tracker deu uma posição a até --tol px do seu clique
  Achou longe   o tracker deu uma posição, mas a mais de --tol px do seu clique
  Perdido       o tracker não deu bola
Nos frames em que você NÃO vê a bola, "falso alarme" é o tracker dizer que tem bola.
Precisão, recall e F1 seguem a convenção de detecção: acerto = verdadeiro positivo; "achou longe" e
falso alarme contam como falso positivo; "achou longe" e "perdido" contam como falso negativo.

Frames "uncertain" e "skip" ficam de fora de tudo. As fatias B e C separam os frames em que o ATD
perde a bola dos em que ele tem: a fatia B é a que mostra se um detector novo RECUPERA o que o ATD
deixa passar.

Leia com cuidado:
  - Com poucos frames as porcentagens têm margem larga (a margem de cada fatia está no título).
  - Como os trackers são medidos nos mesmos frames, diferenças menores que a margem ainda podem ser
    reais, mas esta tabela não faz teste pareado: não declare vencedor por 1 ou 2 pontos.
  - Erro em pixels não é metros. Escolha --tol olhando o tamanho da bola nos frames anotados.
  - "Falso alarme" pode ser uma bola escondida que o tracker estima, não necessariamente erro.

Saída: tabela no terminal e em tracking_bola/out/annotations/tracker_comparison.md (formato
Markdown, pronto para colar no documento técnico em Quarto). Fica fora do git (pasta out/).
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, require, source_frames  # noqa: E402
from load_homography import load as load_homography  # noqa: E402
from check_projection import read_ball  # noqa: E402
from eval_atd import read_annotations, atd_pixel  # noqa: E402

DEFAULT_ANNOTATIONS = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
DEFAULT_OUT = os.path.join(HERE, "out", "annotations", "tracker_comparison.md")
LABEL_WIDTH = 50


def load_tracker(path):
    """{frame: (x, y)} de um CSV frame,x,y[,conf]; várias linhas por frame: vale a de maior conf."""
    if not os.path.isfile(path):
        sys.exit("não achei o CSV do tracker: {}".format(path))
    best = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        for column in ("frame", "x", "y"):
            if column not in fields:
                sys.exit("{}: falta a coluna '{}' (esperado: frame,x,y[,conf])".format(path, column))
        has_conf = "conf" in fields
        for row in reader:
            try:
                x, y = float(row["x"]), float(row["y"])
                frame = int(float(row["frame"]))
                conf = float(row["conf"]) if has_conf and row["conf"] != "" else 0.0
            except ValueError:
                continue
            if np.isnan(x) or np.isnan(y):
                continue
            if frame not in best or conf > best[frame][2]:
                best[frame] = (x, y, conf)
    return {frame: (v[0], v[1]) for frame, v in best.items()}


def distances(preds, truth, frames):
    """Distâncias (px) nos frames em que o tracker deu posição."""
    found = []
    for frame in frames:
        guess = preds.get(frame)
        if guess is not None:
            found.append(np.hypot(guess[0] - truth[frame][0], guess[1] - truth[frame][1]))
    return np.array(found, dtype=float)


def pct(part, whole):
    return "{:.1f}%".format(100.0 * part / whole) if whole else "-"


def px(value):
    return "{:.1f}".format(value)


def margin(n):
    """Margem de erro (95%, pior caso) de uma porcentagem medida em n frames, em pontos."""
    return 98.0 / np.sqrt(n) if n else 0.0


def block(frames, predictions, truth, tol, detail):
    """Linhas de métricas para um conjunto de frames com bola visível."""
    n = len(frames)
    found = [distances(preds, truth, frames) for preds in predictions]
    rows = [("Acerto (até {:g} px)".format(tol), [pct(int((d <= tol).sum()), n) for d in found]),
            ("Achou longe (mais de {:g} px)".format(tol), [pct(int((d > tol).sum()), n) for d in found]),
            ("Perdido (sem bola)", [pct(n - len(d), n) for d in found])]
    if detail:
        rows += [("Acerto até 10 px", [pct(int((d <= 10).sum()), n) for d in found]),
                 ("Acerto até 50 px", [pct(int((d <= 50).sum()), n) for d in found]),
                 ("Erro mediano, só onde achou (px)", [px(np.median(d)) if len(d) else "-" for d in found]),
                 ("Erro p90, só onde achou (px)", [px(np.percentile(d, 90)) if len(d) else "-" for d in found])]
    return rows


def render_text(rows, names):
    column = max(10, max(len(name) for name in names) + 2)
    lines = ["".ljust(LABEL_WIDTH) + "".join(name.rjust(column) for name in names)]
    for kind, label, cells in rows:
        if kind == "head":
            lines += ["", label]
        else:
            lines.append("  " + label.ljust(LABEL_WIDTH - 2) + "".join(cell.rjust(column) for cell in cells))
    return "\n".join(lines)


def render_markdown(rows, names, title):
    lines = [title, "", "| Métrica | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for kind, label, cells in rows:
        if kind == "head":
            lines.append("| **{}** |".format(label) + " |" * len(names))
        else:
            lines.append("| {} | ".format(label) + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracker", action="append", default=[], metavar="NOME=CAMINHO.csv",
                        help="CSV frame,x,y[,conf] de um tracker; pode repetir")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--tol", type=float, default=25.0, help="tolerância de acerto, em pixels (padrão 25)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)

    ball_frames = sorted(f for f, (s, _, _) in annotations.items() if s == "ball" and 0 <= f < n_frames)
    none_frames = sorted(f for f, (s, _, _) in annotations.items() if s == "not_visible" and 0 <= f < n_frames)
    if not ball_frames:
        sys.exit("nenhuma anotação com status 'ball'")
    truth = {f: (annotations[f][1], annotations[f][2]) for f in ball_frames}

    # ATD embutido: só nos frames anotados
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, _, valid = load_homography()
    atd = {}
    for frame in ball_frames + none_frames:
        pixel = atd_pixel(ball, matrices, valid, frame)
        if pixel is not None:
            atd[frame] = pixel

    names, predictions = ["ATD"], [atd]
    for item in args.tracker:
        if "=" not in item:
            sys.exit("--tracker espera NOME=caminho.csv, recebi: {}".format(item))
        name, path = item.split("=", 1)
        preds = load_tracker(path.strip())
        covered = sum(1 for f in ball_frames + none_frames if f in preds)
        print("{}: {} linhas no CSV, {} em frames anotados".format(name.strip(), len(preds), covered))
        if covered == 0:
            print("  ATENÇÃO: nenhum frame anotado tem linha. Confira o índice de frame (começa em 0)")
            print("  e se o tracker rodou nesses frames; senão tudo conta como 'perdido'.")
        names.append(name.strip())
        predictions.append(preds)

    slices = [("A. Todos os frames com bola visível", ball_frames, True),
              ("B. Só onde o ATD NÃO tem bola (o que um detector novo pode recuperar)",
               [f for f in ball_frames if f not in atd], False),
              ("C. Só onde o ATD tem bola", [f for f in ball_frames if f in atd], False)]
    rows = []
    for title, frames, detail in slices:
        if not frames:
            continue
        rows.append(("head", "{} (n = {}, margem até ±{:.0f} pontos)".format(title, len(frames), margin(len(frames))), None))
        rows += [("row", label, cells) for label, cells in block(frames, predictions, truth, args.tol, detail)]

    if none_frames:
        rows.append(("head", "D. Frames em que você NÃO vê a bola (n = {}, margem até ±{:.0f} pontos)".format(
            len(none_frames), margin(len(none_frames))), None))
        rows.append(("row", "Falso alarme (tracker diz ter bola)",
                     [pct(sum(1 for f in none_frames if f in preds), len(none_frames)) for preds in predictions]))

    precision, recall, f1 = [], [], []
    for preds in predictions:
        found = distances(preds, truth, ball_frames)
        tp = int((found <= args.tol).sum())
        fp = (len(found) - tp) + sum(1 for f in none_frames if f in preds)
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(len(ball_frames))
        precision.append("{:.1f}%".format(100 * p))
        recall.append("{:.1f}%".format(100 * r))
        f1.append("{:.1f}%".format(100 * 2 * p * r / (p + r)) if p + r else "0.0%")
    rows.append(("head", "E. Resumo de detecção (tolerância {:g} px; A e D juntos)".format(args.tol), None))
    rows += [("row", "Precisão", precision), ("row", "Recall", recall), ("row", "F1", f1)]

    title = "Comparação de trackers de bola ({} frames com bola, {} sem; tolerância {:g} px)".format(
        len(ball_frames), len(none_frames), args.tol)
    print("\n" + title)
    print(render_text(rows, names))
    print("\nAcerto = posição a até {:g} px do seu clique. Achou longe = deu posição mais distante.".format(args.tol))
    print("Perdido = sem bola. Frames 'uncertain' e 'skip' ficam de fora. Margens: veja os títulos.")

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w") as handle:
        handle.write(render_markdown(rows, names, "**" + title + "**"))
    print("tabela em Markdown: {}".format(args.out))


if __name__ == "__main__":
    main()