"""Tabela de métricas para comparar trackers de bola (ATD, YOLO, WASB, FootAndBall...) contra as
suas anotações à mão.

Colocar em tracking_bola/, junto de eval_atd.py e check_projection.py (este script importa deles).
Só numpy:

    python tracking_bola/compare_trackers.py
    python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo.csv
    python tracking_bola/compare_trackers.py --tracker yolo=... --tracker wasb=... --tol 25

O ATD já vem embutido (bola do ATD projetada em pixels com a homografia). Cada outro tracker é um
CSV com UMA bola por frame, em pixels do vídeo ORIGINAL (se o detector rodou em recortes, converta
de volta antes), no mesmo índice de frame da annotate_ball.py (começa em 0):

    frame,x,y            (coluna opcional: conf)

Frame sem linha no CSV = o tracker disse "sem bola" naquele frame. Por isso rode o tracker em TODOS
os frames anotados, inclusive onde ele não acha nada. Se houver várias linhas no mesmo frame, vale a
de maior conf (ou a primeira, se não houver conf).

Limiar de confiança: com --min-conf 0.3 0.5, cada tracker que tem coluna conf vira duas colunas
(nome@0.3 e nome@0.5), em que o frame só conta como "tem bola" se o melhor candidato tiver conf pelo
menos igual ao limiar. Serve para escolher o limiar sem rodar o detector de novo. O limiar certo é o
que equilibra "perdido" contra "falso alarme"; olhe também a linha F1.

Como cada frame com bola visível é classificado (tolerância --tol, padrão 25 px):
  Acerto        o tracker deu uma posição a até --tol px do seu clique
  Achou longe   o tracker deu uma posição, mas a mais de --tol px do seu clique
  Perdido       o tracker não deu bola
Nos frames em que você NÃO vê a bola, "falso alarme" é o tracker dizer que tem bola.
Precisão, recall e F1 seguem a convenção de detecção: acerto = verdadeiro positivo; "achou longe" e
falso alarme contam como falso positivo; "achou longe" e "perdido" contam como falso negativo.

Linhas de erro em px (seções A e B): só existem onde o tracker deu posição. "Nos acertos" = posições
a até --tol px; "onde achou longe" = posições a mais de --tol px; "todas as posições dadas" = os dois
juntos. Média e mediana de cada grupo; a média pesa muito os erros enormes, a mediana não.

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
    """(melhor, tem_conf, todos) de um CSV frame,x,y[,conf].
    melhor = {frame: (x, y, conf)} com a linha de MAIOR conf de cada frame (é a que vale como a
    posição do tracker); todos = {frame: [(x, y, conf), ...]} com TODOS os candidatos do frame.
    Sem coluna conf, conf vale 0 em todas as linhas e tem_conf é False."""
    if not os.path.isfile(path):
        sys.exit("não achei o CSV do tracker: {}".format(path))
    best, every = {}, {}
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
            every.setdefault(frame, []).append((x, y, conf))
            if frame not in best or conf > best[frame][2]:
                best[frame] = (x, y, conf)
    return {frame: (v[0], v[1], v[2]) for frame, v in best.items()}, has_conf, every


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


def in_candidates(cands, truth, frames, tol):
    """Quantos frames têm ALGUM candidato a até `tol` px do seu clique (não só o de maior conf)."""
    hits = 0
    for frame in frames:
        if any(np.hypot(x - truth[frame][0], y - truth[frame][1]) <= tol for x, y in cands.get(frame, [])):
            hits += 1
    return hits


def block(frames, predictions, candidates, truth, tol, detail):
    """Linhas de métricas para um conjunto de frames com bola visível."""
    n = len(frames)
    found = [distances(preds, truth, frames) for preds in predictions]
    rows = [("Acerto (até {:g} px)".format(tol), [pct(int((d <= tol).sum()), n) for d in found]),
            ("Achou longe (mais de {:g} px)".format(tol), [pct(int((d > tol).sum()), n) for d in found]),
            ("Perdido (sem bola)", [pct(n - len(d), n) for d in found]),
            # Teto do que escolher melhor entre os candidatos poderia render: a bola real está entre
            # eles, mesmo quando não é a de maior confiança? Para o ATD (1 candidato) = Acerto.
            ("Bola entre os candidatos (algum a até {:g} px)".format(tol),
             [pct(in_candidates(cands, truth, frames, tol), n) for cands in candidates])]
    if detail:
        # Erro em px só existe onde o tracker deu uma posição (onde "perdeu" não há distância). Três
        # grupos: os acertos (até --tol), os que achou LONGE (mais de --tol) e todas as posições dadas.
        def stat(function, group):
            return px(function(group)) if len(group) else "-"
        hits = [d[d <= tol] for d in found]
        far = [d[d > tol] for d in found]
        rows += [("Acerto até 10 px", [pct(int((d <= 10).sum()), n) for d in found]),
                 ("Acerto até 50 px", [pct(int((d <= 50).sum()), n) for d in found]),
                 ("Erro médio, nos acertos (px)", [stat(np.mean, d) for d in hits]),
                 ("Erro mediano, nos acertos (px)", [stat(np.median, d) for d in hits]),
                 ("Erro médio, onde achou longe (px)", [stat(np.mean, d) for d in far]),
                 ("Erro mediano, onde achou longe (px)", [stat(np.median, d) for d in far]),
                 ("Erro médio, todas as posições dadas (px)", [stat(np.mean, d) for d in found]),
                 ("Erro mediano, todas as posições dadas (px)", [stat(np.median, d) for d in found]),
                 ("Erro p90, todas as posições dadas (px)", [stat(lambda g: np.percentile(g, 90), d) for d in found])]
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
    # ==========================================================================================
    # AS OPÇÕES DA LINHA DE COMANDO
    #
    # O argparse lê o que você escreve depois de "python tracking_bola/compare_trackers.py" e
    # guarda cada opção num campo de `args` (args.tracker, args.min_conf, args.tol...). Cada
    # parser.add_argument(...) abaixo declara UMA opção. Exemplos de chamada:
    #
    #   python tracking_bola/compare_trackers.py
    #       Só o ATD, que já vem embutido e é sempre a primeira coluna da tabela.
    #
    #   python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo_sports_ball.csv
    #       O ATD contra um detector.
    #
    #   python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo_sports_ball.csv --min-conf 0.05 0.3 0.5 0.7
    #       O mesmo detector em quatro colunas, uma por limiar de confiança.
    #
    #   python tracking_bola/compare_trackers.py --tracker m=...yolo_sports_ball.csv --tracker x=...yolov8x.csv --tol 15
    #       Dois detectores lado a lado, com tolerância de acerto de 15 px.
    # ==========================================================================================
    # description: o primeiro parágrafo do texto no topo do arquivo, em uma linha só
    parser = argparse.ArgumentParser(description=" ".join(__doc__.split("\n\n")[0].split()))

    # --tracker NOME=CAMINHO.csv   (pode aparecer várias vezes)
    #   Um tracker para comparar com o ATD. NOME é o rótulo da coluna na tabela; CAMINHO é o CSV
    #   dele (colunas frame,x,y e, se tiver, conf).
    #   action="append": cada --tracker que você escreve é ACRESCENTADO a uma lista, então dá para
    #   passar quantos quiser. Sem action="append", o último --tracker apagaria os anteriores.
    #   default=[]: sem nenhum --tracker, a lista fica vazia e a tabela tem só o ATD.
    #   metavar: só muda o nome que aparece no --help; não afeta o funcionamento.
    parser.add_argument("--tracker", action="append", default=[], metavar="NOME=CAMINHO.csv",
                        help="tracker a comparar com o ATD (pode repetir)")

    # --min-conf C [C ...]
    #   Um ou mais limiares de confiança, separados por espaço. Cada tracker que tem coluna conf
    #   vira UMA COLUNA POR LIMIAR (yolo@0.3, yolo@0.5...), em que o frame só conta como "tem bola"
    #   se o melhor candidato dele tiver conf >= limiar. Serve para escolher o limiar sem rodar o
    #   detector de novo. Não se aplica ao ATD nem a trackers sem coluna conf.
    #   type=float: o argparse converte o texto "0.3" para o número 0.3.
    #   nargs="+": aceita um ou mais valores depois da opção (0.3, ou 0.3 0.5 0.7...); args.min_conf
    #   vira uma lista de números.
    #   default=None: sem esta opção, args.min_conf é None e NENHUM candidato é descartado.
    parser.add_argument("--min-conf", type=float, nargs="+", default=None, metavar="C",
                        help="limiares de confiança; uma coluna por limiar")

    # --annotations CAMINHO.csv
    #   O CSV de anotações à mão, gerado pela annotate_ball.py. O padrão é o caminho em que a
    #   ferramenta o salva, então normalmente você não precisa escrever esta opção.
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, metavar="CAMINHO.csv",
                        help="CSV de anotações (padrão: o da annotate_ball.py)")

    # --tol PX
    #   Tolerância de acerto, em pixels. Uma posição a até PX do seu clique conta como ACERTO; mais
    #   longe que isso conta como "achou longe". A tabela também mostra os acertos até 10 e 50 px,
    #   para você ver o quanto o resultado depende dessa escolha. Escolha olhando o tamanho da bola
    #   nos frames anotados.
    parser.add_argument("--tol", type=float, default=25.0, metavar="PX",
                        help="tolerância de acerto, em pixels (padrão 25)")

    # --out CAMINHO.md
    #   Onde a tabela também é gravada, em Markdown, para colar no documento técnico (Quarto). O
    #   padrão fica em tracking_bola/out/, que não vai para o Git.
    parser.add_argument("--out", default=DEFAULT_OUT, metavar="CAMINHO.md",
                        help="arquivo Markdown de saída")

    args = parser.parse_args()   # lê a linha de comando; erro de uso interrompe o programa aqui

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

    # A tabela tem uma coluna por "tracker". `names` guarda os rótulos das colunas e `predictions`
    # guarda, na mesma ordem, o que cada tracker previu ({frame: (x, y)}). O ATD é sempre o primeiro.
    # `candidates` acompanha `predictions` coluna a coluna, com TODOS os candidatos de cada frame
    # ({frame: [(x, y), ...]}), e serve para a linha "Bola entre os candidatos". O ATD tem um só.
    names, predictions = ["ATD"], [atd]
    candidates = [{f: [p] for f, p in atd.items()}]
    for item in args.tracker:   # cada item é o texto "NOME=CAMINHO.csv" de um --tracker
        if "=" not in item:
            sys.exit("--tracker espera NOME=caminho.csv, recebi: {}".format(item))
        # split("=", 1) corta só no PRIMEIRO "=": o que vem antes é o nome, o resto é o caminho
        name, path = item.split("=", 1)
        preds, has_conf, every = load_tracker(path.strip())
        name = name.strip()
        covered = sum(1 for f in ball_frames + none_frames if f in preds)
        print("{}: {} frames com bola no CSV, {} em frames anotados".format(name, len(preds), covered))
        if covered == 0:
            print("  ATENÇÃO: nenhum frame anotado tem linha. Confira o índice de frame (começa em 0)")
            print("  e se o tracker rodou nesses frames; senão tudo conta como 'perdido'.")
        # Com --min-conf, este tracker vira uma coluna por limiar: cada coluna mantém só os frames
        # cujo MELHOR candidato tem conf >= limiar (o CSV já guarda o de maior conf por frame).
        # Sem --min-conf (ou se o CSV não tem coluna conf), vira uma coluna só, sem descartar nada.
        if args.min_conf and has_conf:
            for limit in args.min_conf:
                names.append("{}@{:g}".format(name, limit))
                predictions.append({f: (v[0], v[1]) for f, v in preds.items() if v[2] >= limit})
                candidates.append({f: [(x, y) for x, y, c in cs if c >= limit] for f, cs in every.items()})
        else:
            if args.min_conf:
                print("  {} não tem coluna conf: --min-conf não se aplica a ele.".format(name))
            names.append(name)
            predictions.append({f: (v[0], v[1]) for f, v in preds.items()})
            candidates.append({f: [(x, y) for x, y, c in cs] for f, cs in every.items()})

    slices = [("A. Todos os frames com bola visível", ball_frames, True),
              ("B. Só onde o ATD NÃO tem bola (o que um detector novo pode recuperar)",
               [f for f in ball_frames if f not in atd], True),
              ("C. Só onde o ATD tem bola", [f for f in ball_frames if f in atd], False)]
    rows = []
    for title, frames, detail in slices:
        if not frames:
            continue
        rows.append(("head", "{} (n = {}, margem até ±{:.0f} pontos)".format(title, len(frames), margin(len(frames))), None))
        rows += [("row", label, cells)
                 for label, cells in block(frames, predictions, candidates, truth, args.tol, detail)]

    if none_frames:
        rows.append(("head", "D. Frames em que você NÃO vê a bola (n = {}, margem até ±{:.0f} pontos)".format(
            len(none_frames), margin(len(none_frames))), None))
        rows.append(("row", "Falso alarme (tracker diz ter bola)",
                     [pct(sum(1 for f in none_frames if f in preds), len(none_frames)) for preds in predictions]))

    point = []                                   # (precisão, recall) de cada coluna, no limiar da coluna
    for preds in predictions:
        found = distances(preds, truth, ball_frames)
        tp = int((found <= args.tol).sum())
        fp = (len(found) - tp) + sum(1 for f in none_frames if f in preds)
        point.append((tp / float(tp + fp) if tp + fp else 0.0, tp / float(len(ball_frames))))

    def f_score(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    rows.append(("head", "E. Resumo de detecção (tolerância {:g} px; A e D juntos)".format(args.tol), None))
    rows += [("row", "Precisão", ["{:.1f}%".format(100 * p) for p, _ in point]),
             ("row", "Recall", ["{:.1f}%".format(100 * r) for _, r in point]),
             ("row", "F1", ["{:.1f}%".format(100 * f_score(p, r)) for p, r in point])]

    title = "Comparação de trackers de bola ({} frames com bola, {} sem; tolerância {:g} px)".format(
        len(ball_frames), len(none_frames), args.tol)
    print("\n" + title)
    print(render_text(rows, names))
    print("\nAcerto = posição a até {:g} px do seu clique. Achou longe = deu posição mais distante.".format(args.tol))
    print("Perdido = sem bola. Frames 'uncertain' e 'skip' ficam de fora. Margens: veja os títulos.")
    print("'Bola entre os candidatos' = algum candidato (não só o de maior conf) a até {:g} px: mostra o".format(args.tol))
    print("quanto escolher melhor entre os candidatos poderia render. Se for bem maior que 'Acerto', o")
    print("detector vê a bola mas escolhe errado; se for parecido, ele não a vê.")

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(rows, names, "**" + title + "**"))
    print("tabela em Markdown: {}".format(args.out))


if __name__ == "__main__":
    main()