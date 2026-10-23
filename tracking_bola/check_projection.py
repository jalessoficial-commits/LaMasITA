"""B0: a bola do ATD, projetada com a homografia, cai em cima da bola no vídeo?

Colocar em tracking_bola/ (um nível abaixo da raiz, onde está o games.py).
Precisa de numpy, pandas (já usado pelo load_homography) e opencv-python:

    pip install opencv-python
    python tracking_bola/check_projection.py            # 50 frames: metade ao acaso, metade rápidos
    python tracking_bola/check_projection.py --n 30 --seed 1
    python tracking_bola/check_projection.py --frames 12000 34567 50210   # frames escolhidos por você

Por que existe: o treino do detector usa a bola do ATD (coordenadas do campo, 0-1) projetada em
pixels pela homografia como rótulo. Se a projeção não cair sobre a bola, o treino aprende errado.
A dúvida principal é a bola no ar: a homografia vale para o plano do chão. Duas possibilidades:
  1. O ATD guarda a posição "como se a bola estivesse no chão": projetar de volta devolve o pixel
     original, o anel cai sobre a bola mesmo no ar, e os rótulos em pixel estão certos.
  2. O ATD guarda o ponto do chão sob a bola: em bola alta o anel cai deslocado da bola, e os
     rótulos de frames aéreos precisam ser filtrados do treino.
Os frames "rápidos" (maior velocidade de bola no ATD) são provavelmente passes longos e chutões,
onde a bola está alta. Assumo que a 3ª coluna do grupo da bola é a velocidade, como nas trilhas
dos jogadores (m/s), mas isso não está confirmado na documentação.

Saída (em tracking_bola/out/projection_check/, que deve estar no .gitignore):
  contact_sheet.jpg      um recorte por frame, com um anel em volta da posição projetada
  frame_XXXXXX.jpg       o frame inteiro, com a bola projetada e o contorno do campo

Como ler o contact_sheet: o anel deve ENVOLVER a bola, sem cobri-la. Se a bola está dentro do
anel nos frames normais e deslocada nos frames rápidos, é o cenário 2. Se está dentro em todos,
é o cenário 1. Se está fora em todos, a projeção ou o alinhamento de frames está errado; confira
também o contorno do campo nos frames inteiros.
O frame inteiro mostra o contorno do campo (amarelo) e a linha do meio; eles devem coincidir com
as linhas reais do campo.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402

CROP_HALF = 64      # meia-largura do recorte, em pixels do vídeo (recorte de 128 x 128)
SCALE = 2           # ampliação do recorte
COLUMNS = 5
FULL_WIDTH = 1280
FAST_PERCENTILE = 90  # "rápido" = velocidade de bola acima deste percentil
DEFAULT_OUT = os.path.join(HERE, "out", "projection_check")


def read_ball(raw_path, n_frames):
    """(n_frames, 3): x, y (0-1) e velocidade da bola por frame de vídeo; NaN onde ausente."""
    ball = np.full((n_frames, 3), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3 or parts[2][0] == "N":  # "NaN" = bola ausente
                continue
            fields = parts[2].split(",")
            try:
                speed = float(fields[2]) if len(fields) > 2 else np.nan
                ball[int(parts[0]) - 1] = (float(fields[0]), float(fields[1]), speed)
            except (ValueError, IndexError):
                continue
    return ball


def project_many(matrices, x, y):
    """Projeta (x, y) com uma matriz por ponto. matrices (k, 3, 3), x e y (k,)."""
    denominator = matrices[:, 2, 0] * x + matrices[:, 2, 1] * y + matrices[:, 2, 2]
    return ((matrices[:, 0, 0] * x + matrices[:, 0, 1] * y + matrices[:, 0, 2]) / denominator,
            (matrices[:, 1, 0] * x + matrices[:, 1, 1] * y + matrices[:, 1, 2]) / denominator)


def crop_around(cv2, image, cx, cy, half):
    """Recorte (2*half x 2*half) centrado em (cx, cy); preto se o centro estiver fora da imagem."""
    height, width = image.shape[:2]
    x0, y0 = int(round(cx)), int(round(cy))
    if not (0 <= x0 <= width and 0 <= y0 <= height):
        return np.zeros((2 * half, 2 * half, 3), dtype=np.uint8)
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded[y0:y0 + 2 * half, x0:x0 + 2 * half]


def label_tile(cv2, tile, text):
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness, cv2.LINE_AA)
    return tile


def choose_frames(args, ball, indices):
    if args.frames:
        return [(int(f), "manual") for f in args.frames]
    rng = np.random.RandomState(args.seed)
    n_fast = args.n // 2
    random_frames = rng.choice(indices, size=min(args.n - n_fast, len(indices)), replace=False)
    speed = ball[indices, 2]
    fast_pool = np.array([], dtype=indices.dtype)
    if np.isfinite(speed).any():
        threshold = np.nanpercentile(speed, FAST_PERCENTILE)
        fast_pool = np.setdiff1d(indices[speed >= threshold], random_frames)
    fast_frames = rng.choice(fast_pool, size=min(n_fast, len(fast_pool)), replace=False)
    chosen = [(int(f), "ao acaso") for f in random_frames] + [(int(f), "rápido") for f in fast_frames]
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=50, help="quantos frames (metade ao acaso, metade rápidos)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="+", help="frames de vídeo escolhidos por você")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  pip install opencv-python")

    n_frames = source_frames("tactical", GAME)
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    matrices, reprojection_error, valid = load_homography()

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    has_ball = ~np.isnan(ball[:, 0])
    indices = np.flatnonzero(graded & has_ball & valid)

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- estatísticas que não dependem de olhar imagem ---------------------------------------
    px, py = project_many(matrices[indices], ball[indices, 0], ball[indices, 1])
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    print("vídeo: {}x{}, {} frames (esperado {})".format(width, height, video_frames, n_frames))
    if video_frames != n_frames:
        print("  ATENÇÃO: contagem diferente da esperada. Sua decodificação não é a de referência;")
        print("  descubra o porquê antes de confiar em qualquer frame (veja o FAQ.md).")
    print("frames avaliados com bola no ATD e homografia válida: {}".format(len(indices)))
    print("  projetam dentro da imagem: {:.1f}%   fora: {:.1f}%".format(
        100.0 * inside.mean(), 100.0 * (~inside).mean()))
    if (~inside).any():
        print("  Fora da imagem = a bola do ATD não tem lugar no vídeo. Pode ser uma câmera que não")
        print("  cobre o campo inteiro, ou projeção/alinhamento errado.")
    usable_error = reprojection_error[indices]
    print("  erro de reprojeção (menor é melhor): mediana {:.3f}  p99 {:.3f}\n".format(
        float(np.median(usable_error)), float(np.percentile(usable_error, 99))))

    # --- imagens ----------------------------------------------------------------------------
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    chosen = choose_frames(args, ball, indices)
    tiles, misaligned = [], 0
    for frame, kind in chosen:
        if not (0 <= frame < n_frames) or not valid[frame] or np.isnan(ball[frame, 0]):
            print("frame {} ignorado: fora do vídeo, sem homografia válida ou sem bola no ATD".format(frame))
            continue
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            print("frame {} ignorado: o vídeo não devolveu o frame".format(frame))
            continue
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
            misaligned += 1

        x, y = (float(v) for v in project(matrices[frame], ball[frame, 0], ball[frame, 1]))
        speed = ball[frame, 2]

        full = image.copy()
        corners = [project(matrices[frame], cx, cy) for cx, cy in ((0, 0), (1, 0), (1, 1), (0, 1))]
        outline = np.array([[float(a), float(b)] for a, b in corners], dtype=np.float32).round().astype(np.int32)
        cv2.polylines(full, [outline], True, (0, 255, 255), 2, cv2.LINE_AA)
        top = project(matrices[frame], 0.5, 0.0)
        bottom = project(matrices[frame], 0.5, 1.0)
        cv2.line(full, (int(round(float(top[0]))), int(round(float(top[1])))),
                 (int(round(float(bottom[0]))), int(round(float(bottom[1])))), (0, 255, 255), 1, cv2.LINE_AA)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(full, (int(round(x)), int(round(y))), 14, (0, 255, 0), 2, cv2.LINE_AA)
        scale = FULL_WIDTH / float(width)
        full = cv2.resize(full, (FULL_WIDTH, int(round(height * scale))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.out, "frame_{:06d}.jpg".format(frame)), full, [cv2.IMWRITE_JPEG_QUALITY, 90])

        tile = crop_around(cv2, image, x, y, CROP_HALF)
        tile = cv2.resize(tile, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        cv2.circle(tile, (CROP_HALF * SCALE, CROP_HALF * SCALE), 10 * SCALE, (0, 255, 0), 1, cv2.LINE_AA)
        speed_text = "" if np.isnan(speed) else " v={:.0f}".format(speed)
        tiles.append(label_tile(cv2, tile, "{} {}{}".format(frame, kind, speed_text)))

    if not tiles:
        sys.exit("nenhum frame pôde ser desenhado")
    side = 2 * CROP_HALF * SCALE
    rows = (len(tiles) + COLUMNS - 1) // COLUMNS
    sheet = np.zeros((rows * side, COLUMNS * side, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, COLUMNS)
        sheet[row * side:(row + 1) * side, column * side:(column + 1) * side] = tile
    sheet_path = os.path.join(args.out, "contact_sheet.jpg")
    cv2.imwrite(sheet_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if misaligned:
        print("ATENÇÃO: em {} frames o decodificador não parou na posição pedida. O seek do OpenCV".format(misaligned))
        print("pode estar impreciso neste vídeo; nesse caso as imagens podem estar deslocadas no tempo.")
    print("{} frames desenhados.".format(len(tiles)))
    print("contact sheet: {}".format(sheet_path))
    print("frames inteiros: {}".format(os.path.join(args.out, "frame_*.jpg")))
    print("\nLeia: o anel verde deve envolver a bola. Compare os frames 'ao acaso' com os 'rápido'.")


if __name__ == "__main__":
    main()