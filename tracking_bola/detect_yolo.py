"""YOLO sem treino (zero-shot): acha a bola com a classe "sports ball" do COCO, em ladrilhos.

Colocar em tracking_bola/. Precisa de ultralytics (traz o PyTorch) e opencv-python:

    pip install ultralytics opencv-python
    python tracking_bola/detect_yolo.py                       # nos frames anotados (ball, not_visible, uncertain)
    python tracking_bola/detect_yolo.py --weights yolov8x.pt  # modelo maior (mais lento, em geral melhor)
    python tracking_bola/detect_yolo.py --limit 10            # teste rápido, só 10 frames

Detector de bola do Roboflow (YOLOv8 treinado em futebol da Bundesliga; repositório roboflow/sports,
MIT). Pesos: football-ball-detection.pt, link no examples/soccer/setup.sh do repositório. Ele tem
uma só classe (0 = ball) e o próprio Roboflow roda em ladrilhos de 640x640 com imgsz 640:

    python tracking_bola/detect_yolo.py --weights tracking_bola/pesos/football-ball-detection.pt --class-id 0 --out tracking_bola/out/yolo_roboflow.csv

Depois:
    python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo_sports_ball.csv

Por que ladrilhos: a bola é minúscula num frame 1920x1080. Reduzir o frame inteiro para 640 a
apaga (foi o que fez o único teste que achei do COCO sem ajuste: não detectou nenhuma bola). Aqui o
frame é cortado em ladrilhos de --tile pixels com sobreposição, e cada ladrilho entra no modelo em
resolução nativa (1 pixel do vídeo = 1 pixel do modelo, se --imgsz = --tile). A sobreposição
garante que uma bola cortada na borda de um ladrilho apareça inteira no vizinho.

Saída: tracking_bola/out/yolo_sports_ball.csv com frame,x,y,conf, em pixels do vídeo original e
com o índice de frame da annotate_ball.py (começa em 0). Vários candidatos por frame, do maior para
o menor conf; o compare_trackers.py usa o de maior conf. Frame sem linha = o modelo não achou nada.
Candidatos de ladrilhos vizinhos que caem a menos de --merge px um do outro viram um só.

Limiar --conf (padrão 0.05): baixo de propósito, porque a confiança da bola costuma ser baixa. Um
limiar baixo acha mais bolas e também mais falsos alarmes; a tabela do compare_trackers mostra o
saldo. Se rodar de novo com outro --conf, use --out diferente para não sobrescrever.

Cuidados:
  - O ultralytics baixa os pesos (yolov8m.pt etc.) na primeira execução, para a PASTA ATUAL. Ponha
    *.pt no .gitignore. O ultralytics é AGPL-3.0: confira se serve para o seu repositório e cite
    modelo e versão no documento técnico.
  - Rodar de novo regrava o CSV. Se você interromper com Ctrl+C, o que já foi processado é salvo,
    mas os frames restantes contam como "perdido" na tabela: rode de novo antes de comparar.
  - O script imprime o tempo por frame e a estimativa para os ~158 mil frames avaliados: é o
    número que decide se rodar o jogo inteiro cabe no prazo.
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
DEFAULT_OUT = os.path.join(HERE, "out", "yolo_sports_ball.csv")
GRADED_FRAMES = 158857      # frames avaliados (DATA.md), para a estimativa de tempo
STATUSES = ("ball", "not_visible", "uncertain")


def tile_origins(length, tile, stride):
    """Posições iniciais dos ladrilhos ao longo de um eixo; o último termina exatamente na borda."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    origins.append(length - tile)
    return origins


def split_tiles(image, tile, stride):
    """Lista de ladrilhos (recortes) e de suas origens (x0, y0) no frame inteiro."""
    height, width = image.shape[:2]
    tiles, origins = [], []
    for y0 in tile_origins(height, tile, stride):
        for x0 in tile_origins(width, tile, stride):
            tiles.append(image[y0:y0 + tile, x0:x0 + tile])
            origins.append((x0, y0))
    return tiles, origins


def merge_candidates(candidates, radius):
    """Junta candidatos a menos de `radius` px, mantendo o de maior conf. (x, y, conf) -> mesma lista."""
    kept = []
    for x, y, conf in sorted(candidates, key=lambda c: -c[2]):
        if all(np.hypot(x - kx, y - ky) > radius for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept


def detect_frame(model, image, args):
    """Candidatos (x, y, conf) em pixels do frame inteiro."""
    tiles, origins = split_tiles(image, args.tile, args.stride)
    results = model.predict(tiles, imgsz=args.imgsz, conf=args.conf, classes=[args.class_id],
                            device=args.device, verbose=False)
    found = []
    for result, (x0, y0) in zip(results, origins):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        for (left, top, right, bottom), score in zip(xyxy, conf):
            # as caixas voltam em pixels do ladrilho; soma a origem para ir ao frame inteiro
            found.append((x0 + (left + right) / 2.0, y0 + (top + bottom) / 2.0, float(score)))
    return merge_candidates(found, args.merge)


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="yolov8m.pt", help="pesos do ultralytics (padrão yolov8m.pt)")
    parser.add_argument("--tile", type=int, default=640, help="lado do ladrilho, em pixels (padrão 640)")
    parser.add_argument("--overlap", type=float, default=0.2, help="sobreposição entre ladrilhos (padrão 0.2)")
    parser.add_argument("--imgsz", type=int, default=None, help="tamanho de entrada do modelo (padrão = --tile)")
    parser.add_argument("--conf", type=float, default=0.05, help="confiança mínima (padrão 0.05)")
    parser.add_argument("--class-id", type=int, default=32, help="classe da bola no modelo (COCO: 32 = sports ball)")
    parser.add_argument("--device", default=None, help="cpu, cuda, 0... (padrão: o ultralytics escolhe)")
    parser.add_argument("--merge", type=float, default=6.0, help="junta candidatos a menos disso, em px")
    parser.add_argument("--limit", type=int, default=None, help="processar só os N primeiros frames")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.imgsz = args.imgsz or args.tile
    args.stride = max(1, int(args.tile * (1 - args.overlap)))

    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  pip install opencv-python")
    if model_factory is None:
        try:
            from ultralytics import YOLO as model_factory
        except ImportError:
            sys.exit("ultralytics não está instalado:  pip install ultralytics")

    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}. Rode annotate_ball.py antes.".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    frames = sorted(f for f, (s, _, _) in annotations.items() if s in STATUSES and 0 <= f < n_frames)
    if args.limit:
        frames = frames[:args.limit]
    if not frames:
        sys.exit("nenhum frame anotado para processar")

    model = model_factory(args.weights)
    names = getattr(model, "names", None) or {}
    name = str(names.get(args.class_id))
    print("classe {} do modelo: '{}' (todas: {})".format(args.class_id, name, dict(names)))
    if "ball" not in name.lower():
        print("ATENÇÃO: a classe {} não parece ser a bola. Confira --class-id na lista acima.".format(args.class_id))

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    stride_text = "{} px (sobreposição {:.0f}%)".format(args.stride, 100 * args.overlap)
    print("{} frames, ladrilhos de {} px, passo {}, imgsz {}, conf >= {}".format(
        len(frames), args.tile, stride_text, args.imgsz, args.conf))

    rows, seek_warnings, processed = [], 0, 0
    started = time.time()
    try:
        for count, frame in enumerate(frames, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                print("frame {} não pôde ser lido; pulando".format(frame))
                continue
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame + 1:
                seek_warnings += 1
            for x, y, conf in detect_frame(model, image, args):
                rows.append((frame, x, y, conf))
            processed += 1
            if count % 10 == 0 or count == len(frames):
                elapsed = time.time() - started
                print("  {}/{} frames, {:.1f} s/frame".format(count, len(frames), elapsed / count))
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
    with_ball = len({r[0] for r in rows})
    print("\n{} frames processados, {} com algum candidato ({} candidatos no total)".format(
        processed, with_ball, len(rows)))
    if processed:
        per_frame = elapsed / processed
        print("tempo: {:.2f} s/frame. Para os ~{} frames avaliados: ~{:.1f} h nesta máquina.".format(
            per_frame, GRADED_FRAMES, per_frame * GRADED_FRAMES / 3600.0))
    if seek_warnings:
        print("ATENÇÃO: em {} frames o decodificador não parou na posição pedida (seek impreciso).".format(
            seek_warnings))
    print("saída: {}".format(args.out))


if __name__ == "__main__":
    main()