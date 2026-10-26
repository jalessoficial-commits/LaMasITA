"""Ferramenta de anotação da bola: cria o conjunto de validação feito à mão.

Colocar em tracking_bola/ (um nível abaixo da raiz, onde está o games.py). Precisa de numpy e
opencv-python (a versão com janela, não a "headless"):

    pip install opencv-python
    python tracking_bola/annotate_ball.py                 # 300 frames, retoma de onde parou
    python tracking_bola/annotate_ball.py --n 500 --seed 0

Conjunto de TESTE separado (frames 301 a 400 da mesma lista, garantidamente diferentes dos 300 primeiros),
num arquivo próprio, para medir o modelo final em frames que não foram usados para escolher nada:

    python tracking_bola/annotate_ball.py --start 300 --n 400 --out tracking_bola/out/annotations/ball_annotations_teste.csv

Para que serve: não existe gabarito de posição da bola (o GT público só tem posse, e o ATD é o que
estamos medindo). Os cliques viram esse gabarito, usado só para VALIDAR (e, se quiser, treinar)
modelos. O arquivo submetido continua saindo de um algoritmo, como o RULES.md exige.

Amostragem: aleatória entre todos os frames avaliados (os dois tempos), no máximo um frame por
bloco de --min-gap frames (padrão 125 = 5 s), para não anotar frames quase idênticos. A ordem é
determinística pela --seed, então aumentar --n só acrescenta frames no fim da mesma lista.

Como anotar (tudo por teclado e mouse; teclas sem colchetes, por causa do teclado ABNT2):
  Tela inteira:
    clique na bola        abre um recorte ampliado em volta do clique
    v                     bola NÃO visível neste frame
    s                     pular (não é guardado como bola)
    z                     desfazer a última anotação e voltar para aquele frame
    q ou Esc              salvar e sair
  Recorte ampliado:
    clique                posiciona (ou reposiciona) a marca; o CENTRO do anel deve ficar no centro da bola
    Enter ou espaço       confirmar como bola
    m                     confirmar como INCERTO (acho que é a bola, mas não tenho certeza)
    r                     voltar para a tela inteira
  Nas duas telas:
    a / d                 ver o frame anterior / seguinte (a bola em movimento se destaca)
    f                     voltar ao frame que está sendo anotado
  Cliques só valem no frame atual (tecla f), nunca nos vizinhos.

Saída: tracking_bola/out/annotations/ball_annotations.csv, regravado a cada anotação.
  frame, status (ball | uncertain | not_visible | skip), x, y, half, time
x e y são pixels do vídeo na resolução original, coordenada contínua (o pixel i cobre [i, i+1)).
Essa pasta está em out/, que deve estar no .gitignore: são dados derivados do vídeo da competição.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402

try:
    import cv2
except ImportError:  # a mensagem de erro sai em main()
    cv2 = None

NEIGHBORS = 4            # frames vizinhos carregados de cada lado
FINE_HALF = 50           # meia-largura do recorte ampliado, em pixels do vídeo
FINE_ZOOM = 8            # ampliação do recorte
RING_RADIUS = 14         # raio do anel de marcação, em pixels da tela ampliada
BANNER_HEIGHT = 74       # altura da faixa de texto acima da imagem (3 linhas)
COLUMNS = ["frame", "status", "x", "y", "half", "time"]
DEFAULT_OUT = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
WINDOW = "annotate_ball"
CONFIRM_KEYS = (13, 10, 32)   # Enter e espaço


# --- amostragem e arquivo ---------------------------------------------------------------------

def half_of(frame):
    for half, first, last in half_video_blocks(GAME):
        if first <= frame <= last:
            return half
    return 0


def sample_frames(n_frames, min_gap, seed):
    """Lista determinística de frames: um por bloco de min_gap frames, blocos embaralhados."""
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    indices = np.flatnonzero(graded)
    blocks = np.unique(indices // min_gap)
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)
    order = []
    for block in blocks:
        low = np.searchsorted(indices, block * min_gap)
        high = np.searchsorted(indices, (block + 1) * min_gap)
        order.append(int(rng.choice(indices[low:high])))
    return order


def read_annotations(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_annotations(path, rows):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


# --- vídeo e desenho --------------------------------------------------------------------------

def load_window(capture, frame, n_frames):
    """({offset: imagem}, seek_ok) dos frames de frame-NEIGHBORS a frame+NEIGHBORS."""
    first = max(0, frame - NEIGHBORS)
    last = min(n_frames - 1, frame + NEIGHBORS)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    images, seek_ok = {}, True
    for index in range(first, last + 1):
        ok, image = capture.read()
        if not ok:
            break
        if index == first:
            seek_ok = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == first + 1
        images[index - frame] = image
    return images, seek_ok


def crop_around(image, cx, cy, half):
    """(recorte 2*half x 2*half, origem x, origem y). O centro é limitado à imagem."""
    height, width = image.shape[:2]
    x0 = int(round(min(max(cx, 0), width - 1)))
    y0 = int(round(min(max(cy, 0), height - 1)))
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[y0:y0 + 2 * half, x0:x0 + 2 * half], x0 - half, y0 - half


def with_banner(view, lines):
    """Imagem com uma faixa preta de texto ACIMA dela (não por cima, para não esconder a bola).
    Só ASCII: o putText do OpenCV não desenha acentos. Os cliques descontam BANNER_HEIGHT."""
    band = np.zeros((BANNER_HEIGHT, view.shape[1], 3), dtype=np.uint8)
    for number, text in enumerate(lines[:3]):
        cv2.putText(band, text, (6, 20 + 22 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, view])


def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)


def annotate_frame(images, title, view_scale, state):
    """Mostra um frame e devolve ('ball'|'uncertain', x, y), ('not_visible',), ('skip',),
    ('undo',) ou ('quit',)."""
    mode, offset = "coarse", 0
    center = origin = marker = None
    message = ""
    state["click"] = None
    while True:
        where = "frame atual" if offset == 0 else "vizinho {:+d} (cliques desligados)".format(offset)
        if mode == "coarse":
            view = cv2.resize(images[offset], None, fx=view_scale, fy=view_scale,
                              interpolation=cv2.INTER_AREA)
            lines = ["{}  |  {}".format(title, where),
                     "clique=bola  v=nao visivel  s=pular  z=desfazer  q=sair  a/d=vizinhos  f=frame atual"]
        else:
            crop, ox, oy = crop_around(images[offset], center[0], center[1], FINE_HALF)
            origin = (ox, oy)
            view = cv2.resize(crop, None, fx=FINE_ZOOM, fy=FINE_ZOOM, interpolation=cv2.INTER_CUBIC)
            if marker is not None:
                cv2.circle(view, (int(round((marker[0] - ox) * FINE_ZOOM)),
                                  int(round((marker[1] - oy) * FINE_ZOOM))),
                           RING_RADIUS, (0, 255, 0), 1, cv2.LINE_AA)
            lines = ["{}  |  {}".format(title, where),
                     "clique=posicionar  Enter=confirmar  m=incerto  r=voltar  v=nao visivel  a/d  f"]
        if message:
            lines.append(message)
        cv2.imshow(WINDOW, with_banner(view, lines))

        key = cv2.waitKey(20) & 0xFF
        char = chr(key).lower() if key < 128 else ""
        click, state["click"] = state["click"], None
        if click is not None:
            click = (click[0], click[1] - BANNER_HEIGHT)   # coordenadas da imagem, sem a faixa
            if click[1] < 0:
                click = None                                # clique na faixa de texto

        if char == "q" or key == 27:
            return ("quit",)
        if char == "v":
            return ("not_visible",)
        if char == "s":
            return ("skip",)
        if char == "z":
            return ("undo",)
        if char == "a" and (offset - 1) in images:
            offset, message = offset - 1, ""
        elif char == "d" and (offset + 1) in images:
            offset, message = offset + 1, ""
        elif char == "f":
            offset, message = 0, ""
        elif mode == "fine" and char == "r":
            mode, marker, message = "coarse", None, ""
        elif mode == "fine" and marker is not None and (key in CONFIRM_KEYS or char == "m"):
            return ("uncertain" if char == "m" else "ball", marker[0], marker[1])

        if click is not None:
            if offset != 0:
                message = "clique ignorado: volte ao frame atual (tecla f)"
            elif mode == "coarse":
                center = (click[0] / view_scale, click[1] / view_scale)
                mode, marker, message = "fine", None, ""
                _, ox, oy = crop_around(images[0], center[0], center[1], FINE_HALF)
                origin = (ox, oy)
            else:
                marker = (origin[0] + click[0] / float(FINE_ZOOM), origin[1] + click[1] / float(FINE_ZOOM))
                message = ""


# --- programa ---------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=300, help="quantos frames na lista (padrão 300)")
    parser.add_argument("--start", type=int, default=0,
                        help="começa na posição N da lista (padrão 0). Ex.: --start 300 --n 400 = os 100 frames seguintes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-gap", type=int, default=125, help="distância mínima entre frames, em frames")
    parser.add_argument("--view-width", type=int, default=1400, help="largura da tela inteira, em pixels")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if cv2 is None:
        sys.exit("opencv não está instalado:  pip install opencv-python")

    n_frames = source_frames("tactical", GAME)
    if not 0 <= args.start < args.n:
        sys.exit("--start deve estar entre 0 e --n - 1")
    order = sample_frames(n_frames, args.min_gap, args.seed)[args.start:args.n]
    rows = read_annotations(args.out)
    done = {int(row["frame"]) for row in rows}

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if video_frames != n_frames:
        print("ATENÇÃO: o vídeo tem {} frames, esperado {}. Sua decodificação não é a de referência;".format(
            video_frames, n_frames))
        print("as anotações ficariam presas a frames errados. Veja o FAQ.md antes de continuar.")
    view_scale = min(1.0, args.view_width / float(width))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    state = {"click": None}
    cv2.setMouseCallback(WINDOW, on_mouse, state)

    seek_warnings = 0
    position = 0
    print("{} frames na lista, {} já anotados. Teclas: veja o cabeçalho do arquivo.".format(
        len(order), len([f for f in order if f in done])))
    while position < len(order):
        frame = order[position]
        if frame in done:
            position += 1
            continue
        images, seek_ok = load_window(capture, frame, n_frames)
        if 0 not in images:
            print("frame {} não pôde ser lido; pulando".format(frame))
            position += 1
            continue
        if not seek_ok:
            seek_warnings += 1
        print("[{}/{}] frame {}".format(position + 1, len(order), frame))
        result = annotate_frame(images, "[{}/{}] frame {}".format(position + 1, len(order), frame),
                                view_scale, state)
        kind = result[0]
        if kind == "quit":
            break
        if kind == "undo":
            if not rows:
                print("nada para desfazer")
                continue
            last = rows.pop()
            done.discard(int(last["frame"]))
            write_annotations(args.out, rows)
            print("desfeito: frame {} ({})".format(last["frame"], last["status"]))
            if int(last["frame"]) in order:
                position = order.index(int(last["frame"]))
            continue
        has_point = kind in ("ball", "uncertain")
        rows.append({
            "frame": frame, "status": kind,
            "x": "{:.1f}".format(result[1]) if has_point else "",
            "y": "{:.1f}".format(result[2]) if has_point else "",
            "half": half_of(frame), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        done.add(frame)
        write_annotations(args.out, rows)
        position += 1

    cv2.destroyAllWindows()
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("\nanotações salvas em {}".format(args.out))
    print("  " + ", ".join("{}: {}".format(k, counts[k]) for k in sorted(counts)) if counts else "  (vazio)")
    if seek_warnings:
        print("ATENÇÃO: em {} frames o decodificador não parou na posição pedida. O seek do OpenCV pode".format(
            seek_warnings))
        print("estar impreciso neste vídeo, e as anotações desses frames podem estar deslocadas no tempo.")


if __name__ == "__main__":
    main()