"""Mede o tamanho da bola no vídeo para escolher a tolerância (--tol) do compare_trackers.py.

Colocar em tracking_bola/, junto de annotate_ball.py (este script usa o arquivo de anotações dele).
Precisa de opencv-python:

    python tracking_bola/measure_ball.py              # mede 20 frames onde você já marcou a bola
    python tracking_bola/measure_ball.py --n 30       # mais frames
    python tracking_bola/measure_ball.py --summary    # só mostra o resumo, sem abrir a janela
    python tracking_bola/measure_ball.py --bolas 2.5  # outra regra: tolerância = 2,5 diâmetros

Como funciona: para cada frame que você marcou como "ball" no annotate_ball.py, o script abre um
recorte AMPLIADO ao redor da bola (12x). Você faz dois cliques:

    clique 1   no CENTRO da bola
    clique 2   na BORDA da bola (o círculo que aparece deve encostar na borda, por fora da parte nítida;
               não inclua o rastro de movimento, só o corpo da bola)

Enter confirma; depois do clique 2 o círculo fica fixo. Teclas:
    Enter ou espaço   confirma a medida e vai para o próximo frame
    r                 refaz os cliques deste frame
    s                 pula o frame (bola borrada, escondida ou duvidosa: não meça)
    z                 desfaz a medida anterior e volta a ela
    q ou Esc          sai (o que você já mediu fica salvo)
Quatro risquinhos nas margens do recorte apontam para o centro da imagem, onde a bola deve estar.

Resultado: tracking_bola/out/annotations/ball_size.csv (frame,x,y,radius_px,diameter_px) e um resumo
com o diâmetro mediano e a tolerância sugerida. A regra padrão é: tolerância = 5 diâmetros de bola
(--bolas muda o 5). A tolerância é o RAIO em torno do centro verdadeiro dentro do qual o tracker conta
como acerto, que é o que o compare_trackers.py usa. Se você pensou em "5 raios de bola", use --bolas 2.5.

Valores em pixels do vídeo original. Rodar de novo continua de onde parou (frames já medidos ficam).
Essa pasta (out/) deve estar no .gitignore: são dados derivados do vídeo da competição.
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
from games import require  # noqa: E402
from eval_atd import read_annotations  # noqa: E402

try:
    import cv2
except ImportError:  # a mensagem de erro sai em main()
    cv2 = None

HALF = 32                 # meia-largura do recorte, em pixels do vídeo
ZOOM = 12                 # ampliação (recorte de 64 px vira 768 px na tela)
BANNER_HEIGHT = 74        # faixa de texto acima da imagem; os cliques descontam isso
TICK = (170, 215)         # distância (px da tela) do centro em que ficam os risquinhos
COLUMNS = ["frame", "x", "y", "radius_px", "diameter_px"]
DEFAULT_ANNOTATIONS = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
DEFAULT_OUT = os.path.join(HERE, "out", "annotations", "ball_size.csv")
WINDOW = "measure_ball"
CONFIRM_KEYS = (13, 10, 32)   # Enter e espaço


def read_sizes(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as handle:
        return [{"frame": int(r["frame"]), "x": float(r["x"]), "y": float(r["y"]),
                 "radius_px": float(r["radius_px"]), "diameter_px": float(r["diameter_px"])}
                for r in csv.DictReader(handle)]


def write_sizes(path, rows):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"frame": row["frame"], "x": "{:.2f}".format(row["x"]), "y": "{:.2f}".format(row["y"]),
                             "radius_px": "{:.2f}".format(row["radius_px"]),
                             "diameter_px": "{:.2f}".format(row["diameter_px"])})
    os.replace(temporary, path)


def sample_ball_frames(annotations, n, seed):
    """n frames 'ball' escolhidos de forma determinística, espalhados no tempo (ordem embaralhada)."""
    frames = sorted(f for f, (status, _, _) in annotations.items() if status == "ball")
    rng = np.random.RandomState(seed)
    if len(frames) > n:
        picked = rng.choice(len(frames), size=n, replace=False)
        frames = [frames[i] for i in sorted(picked)]
    rng.shuffle(frames)
    return [int(f) for f in frames]


def summarize(rows, factor, reference_tol=25.0):
    """Texto do resumo: diâmetro típico e tolerância sugerida = factor x diâmetro mediano."""
    if not rows:
        return "nenhuma medida ainda"
    diameters = np.array([r["diameter_px"] for r in rows])
    median = float(np.median(diameters))
    q1, q3 = np.percentile(diameters, [25, 75])
    suggested = int(round(factor * median))
    lines = ["{} frames medidos. Diâmetro da bola (px do vídeo original):".format(len(rows)),
             "  mediano {:.1f}   quartis {:.1f} a {:.1f}   mínimo {:.1f}   máximo {:.1f}".format(
                 median, q1, q3, diameters.min(), diameters.max()),
             "Tolerância sugerida = {:g} diâmetros x {:.1f} px = {} px".format(factor, median, suggested),
             "Para comparação: {:g} px (a tolerância padrão) = {:.1f} diâmetros de bola.".format(
                 reference_tol, reference_tol / median),
             "Use:  python tracking_bola/compare_trackers.py --tol {} ...".format(suggested)]
    if len(rows) < 10:
        lines.append("ATENÇÃO: poucas medidas ({}); meça pelo menos 15 a 20 para o mediano valer.".format(len(rows)))
    return "\n".join(lines)


def with_banner(view, lines):
    """Faixa preta de texto ACIMA da imagem (não por cima, para não esconder a bola). Só ASCII."""
    band = np.zeros((BANNER_HEIGHT, view.shape[1], 3), dtype=np.uint8)
    for number, text in enumerate(lines[:3]):
        cv2.putText(band, text, (6, 20 + 22 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, view])


def crop_around(image, cx, cy, half):
    """(recorte 2*half x 2*half, origem x, origem y). Preenche de preto se passar da borda."""
    height, width = image.shape[:2]
    x0 = int(round(min(max(cx, 0), width - 1)))
    y0 = int(round(min(max(cy, 0), height - 1)))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[y0:y0 + 2 * half, x0:x0 + 2 * half], x0 - half, y0 - half


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        state["move"] = (x, y)


def measure_frame(image, cx, cy, title, state):
    """Mostra o recorte e devolve ('ok', raio_px_original, centro_x, centro_y), ('skip',), ('undo',)
    ou ('quit',). centro_x e centro_y são o centro que você clicou, em pixels do vídeo."""
    crop, ox, oy = crop_around(image, cx, cy, HALF)
    base = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    middle = base.shape[0] // 2
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(base, (middle + dx * TICK[0], middle + dy * TICK[0]), (middle + dx * TICK[1], middle + dy * TICK[1]),
                 (0, 255, 255), 1, cv2.LINE_AA)
    center = edge = None
    state["click"] = state["move"] = None
    while True:
        view = base.copy()
        radius_view = None
        if center is not None:
            cv2.circle(view, (int(round(center[0])), int(round(center[1]))), 2, (0, 255, 0), -1, cv2.LINE_AA)
            if edge is not None:
                target = edge
            elif state["move"] is not None:                          # o movimento vem em coordenadas da janela
                target = (state["move"][0], state["move"][1] - BANNER_HEIGHT)
            else:
                target = None
            if target is not None:
                radius_view = float(np.hypot(target[0] - center[0], target[1] - center[1]))
                cv2.circle(view, (int(round(center[0])), int(round(center[1]))), int(round(radius_view)),
                           (0, 255, 0), 1, cv2.LINE_AA)
        if center is None:
            lines = [title, "clique 1 = CENTRO da bola   s=pular  z=desfazer anterior  q=sair",
                     "risquinhos amarelos apontam para onde a bola deve estar"]
        elif edge is None:
            lines = [title, "clique 2 = BORDA da bola (o circulo deve encostar nela)   r=refazer",
                     "s=pular  z=desfazer anterior  q=sair"]
        else:
            lines = [title, "diametro {:.1f} px. Enter=confirmar   r=refazer   s=pular".format(
                2 * radius_view / ZOOM), "z=desfazer anterior  q=sair"]
        cv2.imshow(WINDOW, with_banner(view, lines))

        key = cv2.waitKey(20) & 0xFF
        char = chr(key).lower() if key < 128 else ""
        click, state["click"] = state["click"], None
        if click is not None:
            click = (click[0], click[1] - BANNER_HEIGHT)
            if click[1] < 0:
                click = None                                          # clique na faixa de texto
        if char == "q" or key == 27:
            return ("quit",)
        if char == "s":
            return ("skip",)
        if char == "z":
            return ("undo",)
        if char == "r":
            center = edge = None
            continue
        if edge is not None and key in CONFIRM_KEYS and radius_view and radius_view > 0:
            return ("ok", radius_view / ZOOM, ox + center[0] / ZOOM, oy + center[1] / ZOOM)
        if click is not None:
            if center is None:
                center = click
            elif edge is None:
                edge = click
            else:                                                      # terceiro clique: recomeça
                center, edge = click, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20, help="quantos frames medir (padrão 20)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bolas", type=float, default=5.0,
                        help="tolerância sugerida = este número x diâmetro mediano (padrão 5)")
    parser.add_argument("--summary", action="store_true", help="só mostra o resumo e sai")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    rows = read_sizes(args.out)
    if args.summary:
        print(summarize(rows, args.bolas))
        return
    if cv2 is None:
        sys.exit("opencv não está instalado:  python -m pip install opencv-python")
    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)
    order = sample_ball_frames(annotations, args.n, args.seed)
    if not order:
        sys.exit("nenhum frame 'ball' nas anotações")

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    state = {"click": None, "move": None}
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    measured = {r["frame"] for r in rows}
    position = 0
    print("{} frames na lista, {} já medidos. Teclas: veja o cabeçalho do arquivo.".format(
        len(order), len([f for f in order if f in measured])))
    while position < len(order):
        frame = order[position]
        if frame in measured:
            position += 1
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            print("frame {} não pôde ser lido; pulando".format(frame))
            position += 1
            continue
        _, x, y = annotations[frame]
        title = "frame {} ({}/{} medidos)".format(frame, len(rows), args.n)
        result = measure_frame(image, x, y, title, state)
        if result[0] == "quit":
            break
        if result[0] == "skip":
            position += 1
        elif result[0] == "undo":
            if rows:
                last = rows.pop()
                measured.discard(last["frame"])
                write_sizes(args.out, rows)
                position = max(0, order.index(last["frame"])) if last["frame"] in order else position
        else:
            _, radius, center_x, center_y = result
            rows.append({"frame": frame, "x": center_x, "y": center_y,
                         "radius_px": radius, "diameter_px": 2 * radius})
            measured.add(frame)
            write_sizes(args.out, rows)
            position += 1
    cv2.destroyAllWindows()
    print("\n" + summarize(rows, args.bolas))
    print("medidas salvas em {}".format(args.out))


if __name__ == "__main__":
    main()