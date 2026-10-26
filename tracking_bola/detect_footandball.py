"""FootAndBall (Komorowski et al., VISAPP 2020) nos frames que você anotou, sem treinar nada.

Colocar em tracking_bola/, junto de eval_atd.py (este script importa dele). Precisa de torch, torchvision
e opencv (os dois primeiros você já tem por causa do WASB).

Preparação (uma vez), dentro de tracking_bola/third_party/ (o repositório e os pesos são MIT):

    git clone https://github.com/jac99/FootAndBall tracking_bola/third_party/FootAndBall
    git -C tracking_bola/third_party/FootAndBall rev-parse HEAD > tracking_bola/third_party/FootAndBall_commit.txt

Os pesos já vêm no clone: models/model_20201019_1416_final.pth (825 KB). Para seguir a convenção da pasta
pesos/, copie-os para tracking_bola/pesos/ e aponte --weights para lá. O default é o do clone.

    python tracking_bola/detect_footandball.py --fb-dir tracking_bola/third_party/FootAndBall
    python tracking_bola/compare_trackers.py --tracker atd=... --tracker fb=tracking_bola/out/footandball.csv

O que ele faz de diferente do WASB: a rede é totalmente convolucional e foi feita para o frame em resolução
cheia (o artigo mede a velocidade em 1920x1080; o mapa da bola sai em 1/4 da entrada, 480x270). Não há
ladrilhos nem trio de frames: um frame entra, um mapa de confiança da bola sai, e os picos viram candidatos
(x, y, conf), em pixels do vídeo original. O código original só devolve o melhor pico acima de 0,7; aqui
ficam TODOS os picos acima de --score-threshold (padrão 0.2), como no WASB, para o compare_trackers e a fusão
usarem o conjunto de candidatos. conf = probabilidade (softmax) da célula, entre 0 e 1.

Parâmetros que importam:
  --score-threshold  limiar de confiança dos candidatos (padrão 0.2; o original usa 0.7).
  --scale            encolhe o frame antes da rede (padrão 1.0 = nativo). A rede foi treinada com a bola
                     entre 8 e 20 px (no seu vídeo ela mede ~8 px), então o nativo é o certo; use outro valor
                     só para testar. Coordenadas voltam à escala original.
  --bgr              não converte BGR para RGB (só para diagnosticar se o resultado vier ruim).

Mesmos frames e mesmo formato de saída do detect_wasb.py: out/footandball.csv com frame,x,y,conf
(vários candidatos por frame; frame sem linha = nenhum pico acima do limiar).

Limites: um frame só (sem contexto temporal, como o YOLO); treinado só no ISSIA-CNR (um estádio italiano) e
no SoccerPlayerDetection; o artigo não diz como ele se sai em um vídeo diferente, isso o seu conjunto anotado
vai mostrar. O tempo por frame é impresso no final, com a estimativa para os 158.857 frames avaliados.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402
from eval_atd import read_annotations  # noqa: E402

DEFAULT_ANNOTATIONS = os.path.join(HERE, "out", "annotations", "ball_annotations.csv")
DEFAULT_OUT = os.path.join(HERE, "out", "footandball.csv")
DEFAULT_WEIGHTS = os.path.join("models", "model_20201019_1416_final.pth")
GRADED_FRAMES = 158857
STATUSES = ("ball", "not_visible", "uncertain")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)    # ImageNet, como em data/augmentation.py do original
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_CANDIDATES = 100                                        # padrão do original (max_ball_detections)


def to_tensor_array(image_bgr, bgr=False):
    """Frame BGR uint8 (H, W, 3) -> float32 (1, 3, H, W), RGB, /255 e normalização ImageNet.
    É o mesmo que numpy2tensor do original (BGR->RGB, ToTensor, Normalize)."""
    import cv2
    rgb = image_bgr if bgr else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def resize_for_net(image, scale):
    """(imagem para a rede, fator em x, fator em y). scale 1.0 devolve a imagem sem tocar."""
    import cv2
    if abs(scale - 1.0) < 1e-9:
        return image, 1.0, 1.0
    height, width = image.shape[:2]
    new_w, new_h = max(16, int(round(width * scale))), max(16, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, new_w / float(width), new_h / float(height)


def to_original(x, y, sx, sy):
    """Coordenada (índice de pixel, como o OpenCV e as anotações) da imagem da rede -> vídeo original.
    Os centros dos pixels se alinham pelo meio: x_orig = (x + 0.5) / s - 0.5."""
    return (x + 0.5) / sx - 0.5, (y + 0.5) / sy - 0.5


def detect_frame(predict, image, args):
    """[(x, y, conf), ...] em pixels do frame original, do maior para o menor conf."""
    small, sx, sy = resize_for_net(image, args.scale)
    found = []
    for x, y, conf in predict(small):
        if conf < args.score_threshold:
            continue
        ox, oy = to_original(x, y, sx, sy)
        found.append((ox, oy, conf))
    found.sort(key=lambda c: -c[2])
    return found


def load_real_model(args):
    """Constrói a rede do repositório original e carrega os pesos. Devolve predict(imagem BGR) ->
    [(x, y, conf)] no sistema de coordenadas da imagem que entrou."""
    fb_dir = os.path.abspath(args.fb_dir)
    if not os.path.isfile(os.path.join(fb_dir, "network", "footandball.py")):
        sys.exit("não achei {}.\nClone o repositório: git clone https://github.com/jac99/FootAndBall {}".format(
            os.path.join(fb_dir, "network", "footandball.py"), args.fb_dir))
    weights = args.weights or os.path.join(fb_dir, DEFAULT_WEIGHTS)
    weights = os.path.abspath(weights)
    if not os.path.isfile(weights):
        sys.exit("não achei os pesos em {}. Use --weights.".format(weights))
    sys.path.insert(0, fb_dir)
    try:
        import torch
        import network.footandball as footandball
    except ImportError as error:
        sys.exit("faltou uma biblioteca para o FootAndBall ({}). Precisa de torch e torchvision "
                 "(já usados no WASB) e de opencv.".format(error))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # player_threshold > 1 e só 1 caixa de jogador: ignoramos os jogadores, só a bola interessa aqui
    model = footandball.model_factory("fb1", "detect", max_player_detections=1, max_ball_detections=MAX_CANDIDATES,
                                      player_threshold=2.0, ball_threshold=0.0)
    try:
        state = torch.load(weights, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("modelo FootAndBall carregado ({}), dispositivo {}".format(os.path.basename(weights), device))
    label_ball = footandball.BALL_LABEL

    def predict(image_bgr):
        tensor = torch.from_numpy(to_tensor_array(image_bgr, args.bgr)).to(device)
        with torch.no_grad():
            detections = model(tensor)[0]
        keep = detections["labels"] == label_ball
        boxes = detections["boxes"][keep].cpu().numpy()
        scores = detections["scores"][keep].cpu().numpy()
        # a caixa da bola tem tamanho fixo em volta do centro da célula: o centro é a média dos cantos
        return [(float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0), float(s)) for b, s in zip(boxes, scores)]

    return predict


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fb-dir", default=None, help="pasta do repositório FootAndBall clonado")
    parser.add_argument("--weights", default=None, help="pesos .pth (padrão: <fb-dir>/models/model_20201019_1416_final.pth)")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="conf mínima dos candidatos (original 0.7; padrão 0.2)")
    parser.add_argument("--scale", type=float, default=1.0, help="fator de escala do frame antes da rede (padrão 1.0 = nativo)")
    parser.add_argument("--bgr", action="store_true", help="não converter BGR para RGB (diagnóstico)")
    parser.add_argument("--device", default=None, help="cpu ou cuda (padrão: cuda se houver)")
    parser.add_argument("--limit", type=int, default=None, help="processar só os N primeiros frames")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if model_factory is None and not args.fb_dir:
        parser.error("--fb-dir é obrigatório (pasta do FootAndBall clonado)")
    if args.scale <= 0:
        parser.error("--scale deve ser positivo")

    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  python -m pip install opencv-python")
    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    frames = sorted(f for f, (s, _, _) in annotations.items() if s in STATUSES and 0 <= f < n_frames)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit("nenhum frame anotado para processar")

    predict = (model_factory or load_real_model)(args)

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    print("{} frames, limiar {}, escala {}".format(len(frames), args.score_threshold, args.scale))

    rows, processed, previous = [], 0, None
    started = time.time()
    try:
        for count, frame in enumerate(frames, start=1):
            if previous is None or frame != previous + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            previous = frame
            if not ok:
                print("frame {}: não consegui ler; pulando".format(frame))
                previous = None
                continue
            for x, y, conf in detect_frame(predict, image, args):
                rows.append((frame, x, y, conf))
            processed += 1
            if count % 10 == 0 or count == len(frames):
                print("  {}/{} frames, {:.2f} s/frame".format(count, len(frames), (time.time() - started) / count))
    except KeyboardInterrupt:
        print("\ninterrompido: {} de {} frames processados; salvando o que há.".format(processed, len(frames)))

    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "conf"])
        for frame, x, y, conf in sorted(rows, key=lambda r: (r[0], -r[3])):
            writer.writerow([frame, "{:.2f}".format(x), "{:.2f}".format(y), "{:.4f}".format(conf)])

    elapsed = time.time() - started
    print("\n{} frames processados, {} com algum candidato ({} candidatos no total)".format(
        processed, len({r[0] for r in rows}), len(rows)))
    if processed:
        per_frame = elapsed / processed
        print("tempo: {:.2f} s por frame. Jogo inteiro ({} frames): ~{:.1f} h nesta máquina.".format(
            per_frame, GRADED_FRAMES, per_frame * GRADED_FRAMES / 3600.0))
    print("saída: {}".format(args.out))


if __name__ == "__main__":
    main()