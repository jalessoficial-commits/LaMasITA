"""WASB-SBDT (HRNet, pesos de futebol do autor) nos frames anotados. Mesmo formato de saída do detect_yolo.py.

Colocar em tracking_bola/. O WASB NÃO vai dentro do seu repositório: clone-o em outra pasta (o código é
MIT; os pesos estão no Google Drive do autor) e aponte para ela com --wasb-dir.

    # uma vez, FORA do repositório do hackathon (ex.: C:\\Users\\jales\\WASB-SBDT):
    git clone https://github.com/nttcom/WASB-SBDT
    #   baixe os pesos de futebol (WASB, "wasb_soccer_best.pth.tar") do link da tabela de
    #   MODEL_ZOO.md do repositório e ponha em WASB-SBDT/pretrained_weights/
    python -m pip install torch hydra-core omegaconf opencv-python

    python tracking_bola/detect_wasb.py --wasb-dir C:\\Users\\jales\\WASB-SBDT --limit 10   # teste rápido
    python tracking_bola/detect_wasb.py --wasb-dir C:\\Users\\jales\\WASB-SBDT

Depois:
    python tracking_bola/compare_trackers.py --tracker yolo=tracking_bola/out/yolo_sports_ball.csv ^
        --tracker wasb=tracking_bola/out/wasb.csv

Como o WASB funciona (e por que é diferente do YOLO): em vez de olhar um frame só, ele recebe 3 frames
CONSECUTIVOS empilhados (9 canais, 288x512) e devolve um mapa de calor por frame, onde a bola é um pico.
Ele foi treinado em vídeo de futebol, então já sabe como a bola de futebol aparece e usa o movimento
(a bola se mexe, a chuteira parada e a marcação do campo não) para separá-la do resto. A bola vira um
pico no mapa de calor. A entrada do modelo é de 512x288 pixels, bem menor que o frame de 1920x1080,
por isso o frame é cortado em ladrilhos (veja a ATENÇÃO abaixo).

ATENÇÃO (medido com diagnose_wasb.py): com o frame inteiro encolhido a 512x288 o modelo NÃO enxerga a bola
deste vídeo (2% de acerto); com recortes em resolução nativa enxerga (40% no melhor caso). Por isso o
frame é cortado em ladrilhos de 512x288 (--scale 1.0, nativo; 25 ladrilhos por frame, mais lento).
--scale 0.5 ou 0.75 encolhe antes de cortar: menos ladrilhos, mais rápido, bola menor. O diagnóstico
com --scales mostra qual escala ainda funciona. (Por isso o texto abaixo, escrito antes da medição,
sobre "o frame inteiro virar 512x288", vale só para --scale 0.267.)

Como cada frame anotado é processado: para o frame f, o script lê os 3 frames consecutivos que o
contêm e usa o mapa de calor do próprio f (--position diz qual dos 3 é o f; o padrão 1 = frame do
meio, que vê passado e futuro). Os frames das pontas do vídeo usam o trio mais próximo.

Saída: tracking_bola/out/wasb.csv com frame,x,y,conf, em pixels do vídeo original (frame começa em 0,
como na annotate_ball.py). Vários candidatos por frame, do maior para o menor conf. Frame sem linha =
nenhum pico passou do limiar. conf = valor máximo do mapa de calor no pico (entre 0 e 1).

Pós-processamento (igual ao do repositório original): sigmoide no mapa de calor, limiar, componentes
conexos, e a posição de cada pico é o centro de massa ponderado pelo mapa de calor. O original usa
limiar 0.5; aqui o padrão é --score-threshold 0.2 para ver mais candidatos, e você filtra depois com
`compare_trackers.py --min-conf 0.5` para reproduzir o comportamento original.

Suposições que NÃO consegui testar com o repositório real (só com um modelo fictício). Se o resultado
vier muito ruim, desconfie delas primeiro:
  - a ordem de cores (o código assume RGB, como as imagens lidas pelo PIL no original; --bgr inverte);
  - a normalização (média/desvio do ImageNet, lida do código do original);
  - a montagem da configuração com o hydra (config "eval" com dataset=soccer e model=wasb, como no
    GET_STARTED.md do repositório) e o formato do checkpoint (chave 'model_state_dict').
O original só roda em GPU CUDA; este script roda também em CPU (--device cpu), mais lento.
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
DEFAULT_OUT = os.path.join(HERE, "out", "wasb.csv")
GRADED_FRAMES = 158857
STATUSES = ("ball", "not_visible", "uncertain")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)    # ImageNet, como no original
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FRAMES_IN = 3
EDGE_MARGIN = 8             # px do ladrilho: picos mais perto que isso de uma borda interna são descartados


def scale_factor(width, height, target_width):
    """O original centra a imagem e escala o LADO MAIOR para a largura do alvo (fator único)."""
    return target_width / float(max(width, height))


def forward_matrix(width, height, input_wh):
    """Matriz 2x3 (cv2.warpAffine) que leva o frame original ao tamanho de entrada do modelo."""
    k = scale_factor(width, height, input_wh[0])
    return np.array([[k, 0.0, input_wh[0] / 2.0 - k * width / 2.0],
                     [0.0, k, input_wh[1] / 2.0 - k * height / 2.0]], dtype=np.float32)


def to_original(x, y, width, height, out_wh):
    """Ponto do mapa de calor (colunas, linhas) -> pixels do frame original."""
    k = scale_factor(width, height, out_wh[0])
    return (x - out_wh[0] / 2.0) / k + width / 2.0, (y - out_wh[1] / 2.0) / k + height / 2.0


def preprocess(frames, input_wh, bgr=False):
    """3 frames BGR do OpenCV -> array (9, altura, largura): cada frame normalizado, empilhados em canais."""
    import cv2
    matrix = forward_matrix(frames[0].shape[1], frames[0].shape[0], input_wh)
    planes = []
    for frame in frames:
        warped = cv2.warpAffine(frame, matrix, tuple(input_wh), flags=cv2.INTER_LINEAR)
        if not bgr:
            warped = warped[:, :, ::-1]                     # BGR -> RGB
        image = (warped.astype(np.float32) / 255.0 - MEAN) / STD
        planes.append(image.transpose(2, 0, 1))             # HWC -> CHW
    return np.concatenate(planes, axis=0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def find_blobs(heatmap, threshold):
    """Picos do mapa de calor: (x, y, pico, massa). Como o original: limiar, componentes conexos,
    centro de massa ponderado pelo mapa de calor."""
    import cv2
    mask = (heatmap > threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(mask)
    blobs = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        weights = heatmap[ys, xs].astype(np.float64)
        mass = weights.sum()
        blobs.append(((xs * weights).sum() / mass, (ys * weights).sum() / mass, float(weights.max()), float(mass)))
    return blobs


def tile_origins(length, tile, stride):
    """Posições iniciais dos ladrilhos ao longo de um eixo; o último termina exatamente na borda."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    origins.append(length - tile)
    return origins


def prepare(frames, scale):
    """Encolhe os frames por `scale` (1.0 = nativo) e devolve (frames, sx, sy), com sx, sy = fator real
    de cada eixo (o tamanho vira inteiro, então sx e sy diferem um pouco de `scale`)."""
    import cv2
    height, width = frames[0].shape[:2]
    if scale != 1.0:
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
    return frames, frames[0].shape[1] / float(width), frames[0].shape[0] / float(height)


def pad_to(frames, size_wh):
    """Completa com preto (embaixo e à direita) até ter pelo menos o tamanho de um ladrilho."""
    import cv2
    height, width = frames[0].shape[:2]
    pad_h, pad_w = max(0, size_wh[1] - height), max(0, size_wh[0] - width)
    if not (pad_h or pad_w):
        return frames
    return [cv2.copyMakeBorder(f, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for f in frames]


def merge_candidates(candidates, radius):
    """Junta candidatos a menos de `radius` px (ladrilhos vizinhos veem a mesma bola), mantendo o de maior conf."""
    kept = []
    for x, y, conf in sorted(candidates, key=lambda c: -c[2]):
        if all(np.hypot(x - kx, y - ky) > radius for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept


def detect_triplet(predict, frames, targets, args, width, height):
    """{alvo: candidatos (x, y, conf)} para os frames `targets` (índices 0..2 dentro de `frames`), em
    pixels originais. Os 3 frames são encolhidos por --scale, cortados em ladrilhos do tamanho de
    entrada do modelo (com sobreposição --tile-overlap), e cada ladrilho vira um trio na entrada. Uma
    só passada do modelo dá o mapa de calor dos 3 frames, então pedir os 3 alvos custa o mesmo que 1."""
    scaled, sx, sy = prepare(frames, args.scale)
    scaled = pad_to(scaled, args.input_wh)
    tile_w, tile_h = args.input_wh
    scaled_h, scaled_w = scaled[0].shape[:2]
    xs = tile_origins(scaled_w, tile_w, max(1, int(tile_w * (1 - args.tile_overlap))))
    ys = tile_origins(scaled_h, tile_h, max(1, int(tile_h * (1 - args.tile_overlap))))
    origins = [(x0, y0) for y0 in ys for x0 in xs]
    found = {target: [] for target in targets}
    for first in range(0, len(origins), args.batch):
        chunk = origins[first:first + args.batch]
        batch = np.stack([preprocess([f[y0:y0 + tile_h, x0:x0 + tile_w] for f in scaled], args.input_wh, args.bgr)
                          for x0, y0 in chunk])
        logits = np.asarray(predict(batch))                  # (ladrilhos, 3, altura_saida, largura_saida)
        for i, (x0, y0) in enumerate(chunk):
            for target in targets:
                heatmap = sigmoid(logits[i, target])
                out_wh = (heatmap.shape[1], heatmap.shape[0])
                for x, y, peak, _mass in find_blobs(heatmap, args.score_threshold):
                    tx, ty = to_original(x, y, tile_w, tile_h, out_wh)      # pixels do ladrilho
                    # Pico colado numa borda INTERNA do ladrilho = bola cortada ao meio, com posição
                    # deslocada. A sobreposição garante que o ladrilho vizinho a vê inteira, então este
                    # candidato é descartado. Nas bordas do frame não há vizinho: ali ele vale.
                    if ((x0 > 0 and tx < EDGE_MARGIN) or (x0 + tile_w < scaled_w and tx > tile_w - 1 - EDGE_MARGIN)
                            or (y0 > 0 and ty < EDGE_MARGIN) or (y0 + tile_h < scaled_h and ty > tile_h - 1 - EDGE_MARGIN)):
                        continue
                    found[target].append(((x0 + tx) / sx, (y0 + ty) / sy, peak))     # pixels do frame original
    return {target: merge_candidates(found[target], args.merge) for target in targets}


def detect_frame(predict, frames, target, args, width, height):
    """Candidatos (x, y, conf) do frame `target` (índice 0..2 dentro de `frames`), em pixels originais."""
    return detect_triplet(predict, frames, [target], args, width, height)[target]


def load_real_model(args):
    """Monta o modelo com o código do repositório do WASB. Devolve (predict, (largura, altura) de entrada)."""
    # o hydra exige caminho absoluto; --wasb-dir e --weights podem vir relativos à pasta atual
    args.wasb_dir = os.path.abspath(args.wasb_dir)
    wasb_src = os.path.join(args.wasb_dir, "src")
    if not os.path.isdir(wasb_src):
        sys.exit("não achei {}. Use --wasb-dir com a pasta do repositório WASB-SBDT clonado.".format(wasb_src))
    weights = os.path.abspath(args.weights or os.path.join(args.wasb_dir, "pretrained_weights", "wasb_soccer_best.pth.tar"))
    if not os.path.isfile(weights):
        sys.exit("não achei os pesos {}. Baixe o 'WASB' de futebol do MODEL_ZOO.md e use --weights.".format(weights))
    try:
        import torch
        from hydra import compose, initialize_config_dir
    except ImportError as erro:
        sys.exit("faltou instalar algo ({}):  python -m pip install torch hydra-core omegaconf".format(erro))
    sys.path.insert(0, wasb_src)
    try:
        from models import build_model
    except ImportError as erro:
        sys.exit("não consegui importar models do WASB ({}). Instale as dependências do repositório.".format(erro))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with initialize_config_dir(config_dir=os.path.join(wasb_src, "configs"), version_base=None):
        cfg = compose(config_name="eval", overrides=["dataset=soccer", "model=wasb"])
    model = build_model(cfg)
    checkpoint = torch.load(weights, map_location=device, weights_only=False)   # arquivo do autor (pickle)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    input_wh = (int(cfg["model"]["inp_width"]), int(cfg["model"]["inp_height"]))
    scale = list(cfg["model"]["out_scales"])[0]
    print("modelo WASB carregado ({}), entrada {}x{}, dispositivo {}".format(weights, input_wh[0], input_wh[1], device))

    def predict(batch):
        with torch.no_grad():
            outputs = model(torch.from_numpy(batch).to(device))
            return outputs[scale].float().cpu().numpy()
    return predict, input_wh


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wasb-dir", default=None, help="pasta do repositório WASB-SBDT clonado")
    parser.add_argument("--weights", default=None, help="pesos .pth.tar (padrão: <wasb-dir>/pretrained_weights/wasb_soccer_best.pth.tar)")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="limiar do mapa de calor (original 0.5; padrão 0.2)")
    parser.add_argument("--position", type=int, default=1, choices=[0, 1, 2], help="posição do frame alvo no trio (padrão 1 = meio)")
    parser.add_argument("--scale", type=float, default=1.0, help="fator de encolhimento do frame antes de cortar em ladrilhos (padrão 1.0 = nativo)")
    parser.add_argument("--tile-overlap", type=float, default=0.2, help="sobreposição entre ladrilhos (padrão 0.2)")
    parser.add_argument("--batch", type=int, default=8, help="ladrilhos por passada do modelo (padrão 8)")
    parser.add_argument("--merge", type=float, default=6.0, help="junta candidatos a menos disso, em px (padrão 6)")
    parser.add_argument("--bgr", action="store_true", help="não converter BGR para RGB (teste se o resultado vier ruim)")
    parser.add_argument("--device", default=None, help="cpu ou cuda (padrão: cuda se houver)")
    parser.add_argument("--limit", type=int, default=None, help="processar só os N primeiros frames")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if model_factory is None and not args.wasb_dir:
        parser.error("--wasb-dir é obrigatório (pasta do WASB-SBDT clonado)")

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

    predict, args.input_wh = (model_factory or load_real_model)(args)

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    print("{} frames, trio de {} frames consecutivos, limiar {}".format(len(frames), FRAMES_IN, args.score_threshold))

    rows, processed, seek_warnings = [], 0, 0
    started = time.time()
    try:
        for count, frame in enumerate(frames, start=1):
            start = max(0, min(frame - args.position, n_frames - FRAMES_IN))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            trio = []
            for _ in range(FRAMES_IN):
                ok, image = capture.read()
                if not ok:
                    break
                trio.append(image)
            if len(trio) < FRAMES_IN:
                print("frame {}: não consegui ler o trio; pulando".format(frame))
                continue
            height, width = trio[0].shape[:2]
            for x, y, conf in detect_frame(predict, trio, frame - start, args, width, height):
                rows.append((frame, x, y, conf))
            processed += 1
            if count % 10 == 0 or count == len(frames):
                print("  {}/{} frames, {:.1f} s/frame".format(count, len(frames), (time.time() - started) / count))
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
        # no jogo inteiro cada passada do modelo serve aos 3 frames do trio, então o custo cai a 1/3
        print("tempo: {:.2f} s por trio. Jogo inteiro (trios sem sobreposição, {} frames): ~{:.1f} h nesta máquina.".format(
            per_frame, GRADED_FRAMES, per_frame * GRADED_FRAMES / 3.0 / 3600.0))
    print("saída: {}".format(args.out))


if __name__ == "__main__":
    main()