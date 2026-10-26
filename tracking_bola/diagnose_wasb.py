"""Diagnóstico do WASB: o modelo enxerga a bola? Em que condição?

Colocar em tracking_bola/ (ao lado do detect_wasb.py, que ele reaproveita). Mesmos argumentos de modelo:

    python tracking_bola/diagnose_wasb.py --wasb-dir tracking_bola/third_party/WASB-SBDT ^
        --weights tracking_bola/pesos/wasb_soccer_best.pth.tar

Para cada frame anotado como "ball" (o clique é a verdade), roda o modelo de 4 jeitos e mede o mapa de calor:

  global-RGB   o frame inteiro encolhido para 512x288 (o que o detect_wasb.py faz)
  global-BGR   o mesmo, com a ordem de cores trocada
  recorte-RGB  um recorte de 512x288 pixels ORIGINAIS centrado na bola que você clicou (sem encolher)
  recorte-BGR  o mesmo, com cores trocadas
  recorte-RGB@0.5, @0.75  (--scales) o frame encolhido por esse fator antes do recorte: a bola fica menor
               e o recorte cobre uma área maior. Mostra até que escala o modelo ainda enxerga a bola.

O recorte é o melhor cenário possível: a bola aparece no tamanho real e está no meio da imagem. Como ler:

  - recorte bom, global ruim  -> problema de ESCALA: a bola encolhida some. Solução: ladrilhos em
    resolução nativa, como no YOLO (um trio por ladrilho de 512x288).
  - RGB bom e BGR ruim (ou o contrário) -> ordem de cores; me diga qual.
  - tudo ruim, inclusive o recorte -> a entrada não bate com o que o modelo espera (normalização,
    ordem dos frames, o tipo de imagem do treino) ou o modelo não serve para este vídeo.
  - global bom aqui mas ruim no detect_wasb.py -> bug na conversão de coordenadas do detect_wasb.py.

Colunas da tabela: "pico" = maior valor do mapa de calor (0 a 1) no frame; "na bola" = maior valor a
até ~3 células da bola real; "acerto" = % de frames em que o ponto mais forte do mapa cai a até 25 px
(originais) do clique. Se "na bola" é alto e "acerto" baixo, o modelo vê a bola mas há outro ponto mais
forte; se "na bola" é perto de 0, ele não a vê.

Também salva imagens em tracking_bola/out/wasb_diag/ (as primeiras --images N): em cima o frame inteiro
encolhido com o mapa de calor (vermelho = forte) e o círculo verde na bola real; embaixo o recorte nativo.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, require, source_frames  # noqa: E402
from eval_atd import read_annotations  # noqa: E402
import detect_wasb as dw  # noqa: E402

DEFAULT_ANNOTATIONS = dw.DEFAULT_ANNOTATIONS
DEFAULT_IMAGES = os.path.join(HERE, "out", "wasb_diag")
CROP_W, CROP_H = 512, 288
NEIGHBORHOOD = 3        # células do mapa de calor ao redor da bola real


def heatmap_of(predict, trio, target, input_wh, bgr):
    batch = dw.preprocess(trio, input_wh, bgr)[None]
    return dw.sigmoid(np.asarray(predict(batch))[0, target])


def measure(heatmap, click_hm):
    """(pico, valor perto da bola, (x, y) do ponto mais forte), em coordenadas do mapa de calor."""
    row, col = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    cx, cy = int(round(click_hm[0])), int(round(click_hm[1]))
    y0, y1 = max(0, cy - NEIGHBORHOOD), min(heatmap.shape[0], cy + NEIGHBORHOOD + 1)
    x0, x1 = max(0, cx - NEIGHBORHOOD), min(heatmap.shape[1], cx + NEIGHBORHOOD + 1)
    near = float(heatmap[y0:y1, x0:x1].max()) if y1 > y0 and x1 > x0 else 0.0
    return float(heatmap.max()), near, (float(col), float(row))


def overlay(image, heatmap, click_px, size):
    import cv2
    base = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    heat = cv2.resize((heatmap * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    mixed = cv2.addWeighted(base, 0.6, colored, 0.4, 0)
    cv2.circle(mixed, (int(round(click_px[0])), int(round(click_px[1]))), 9, (0, 255, 0), 1)
    return mixed


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wasb-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--position", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--frames", type=int, default=40, help="quantos frames 'ball' usar (padrão 40)")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75],
                        help="escalas extras do recorte (padrão 0.5 0.75): o frame é encolhido por esse fator e "
                             "o recorte de 512x288 passa a cobrir uma área maior; 1.0 já é a linha recorte-RGB")
    parser.add_argument("--images", type=int, default=6, help="quantas imagens salvar (padrão 6)")
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    args = parser.parse_args(argv)
    if model_factory is None and not args.wasb_dir:
        parser.error("--wasb-dir é obrigatório")
    import cv2

    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    balls = sorted((f, v[1], v[2]) for f, v in annotations.items() if v[0] == "ball" and 0 <= f < n_frames)
    if not balls:
        sys.exit("nenhum frame 'ball' nas anotações")
    step = max(1, len(balls) // args.frames)
    balls = balls[::step][:args.frames]

    predict, input_wh = (model_factory or dw.load_real_model)(args)
    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    if args.images:
        os.makedirs(args.images_dir, exist_ok=True)

    names = ["global-RGB", "global-BGR", "recorte-RGB", "recorte-BGR"]
    names += ["recorte-RGB@{:g}".format(sc) for sc in args.scales]
    stats = {n: {"peak": [], "near": [], "hit": []} for n in names}
    saved = 0
    for frame, cx, cy in balls:
        start = max(0, min(frame - args.position, n_frames - dw.FRAMES_IN))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        trio = [capture.read()[1] for _ in range(dw.FRAMES_IN)]
        if any(t is None for t in trio):
            continue
        target = frame - start
        height, width = trio[0].shape[:2]
        x0 = int(min(max(round(cx - CROP_W / 2), 0), width - CROP_W))
        y0 = int(min(max(round(cy - CROP_H / 2), 0), height - CROP_H))
        crops = [t[y0:y0 + CROP_H, x0:x0 + CROP_W] for t in trio]
        shown = {}
        for bgr in (False, True):
            tag = "BGR" if bgr else "RGB"
            # global: frame inteiro encolhido
            hm = heatmap_of(predict, trio, target, input_wh, bgr)
            k = dw.scale_factor(width, height, hm.shape[1])
            click_hm = ((cx - width / 2.0) * k + hm.shape[1] / 2.0, (cy - height / 2.0) * k + hm.shape[0] / 2.0)
            peak, near, (px, py) = measure(hm, click_hm)
            ox, oy = dw.to_original(px, py, width, height, (hm.shape[1], hm.shape[0]))
            s = stats["global-" + tag]
            s["peak"].append(peak); s["near"].append(near); s["hit"].append(np.hypot(ox - cx, oy - cy) <= 25)
            if not bgr:
                shown["global"] = (hm, trio[target])
            # recorte nativo centrado na bola
            hm = heatmap_of(predict, crops, target, (CROP_W, CROP_H), bgr)
            k = hm.shape[1] / float(CROP_W)
            click_hm = ((cx - x0) * k, (cy - y0) * k)
            peak, near, (px, py) = measure(hm, click_hm)
            s = stats["recorte-" + tag]
            s["peak"].append(peak); s["near"].append(near)
            s["hit"].append(np.hypot(px / k - (cx - x0), py / k - (cy - y0)) <= 25)
            if not bgr:
                shown["recorte"] = (hm, crops[target], (cx - x0, cy - y0))
        for scale in args.scales:      # recorte centrado na bola, mas com o frame encolhido (mais contexto)
            scaled, sx, sy = dw.prepare(trio, scale)
            scaled = dw.pad_to(scaled, (CROP_W, CROP_H))
            scaled_h, scaled_w = scaled[0].shape[:2]
            ccx, ccy = cx * sx, cy * sy
            sx0 = int(min(max(round(ccx - CROP_W / 2), 0), scaled_w - CROP_W))
            sy0 = int(min(max(round(ccy - CROP_H / 2), 0), scaled_h - CROP_H))
            pieces = [t[sy0:sy0 + CROP_H, sx0:sx0 + CROP_W] for t in scaled]
            hm = heatmap_of(predict, pieces, target, (CROP_W, CROP_H), False)
            k = hm.shape[1] / float(CROP_W)
            peak, near, (px, py) = measure(hm, ((ccx - sx0) * k, (ccy - sy0) * k))
            s = stats["recorte-RGB@{:g}".format(scale)]
            s["peak"].append(peak); s["near"].append(near)
            s["hit"].append(np.hypot(px / k - (ccx - sx0), py / k - (ccy - sy0)) <= 25 * scale)
        if saved < args.images:
            top_hm, top_img = shown["global"]
            kk = dw.scale_factor(width, height, 512)
            top_click = ((cx - width / 2.0) * kk + 256, (cy - height / 2.0) * kk + 144)
            top = overlay(top_img, top_hm, top_click, (512, 288))
            bm, bi, bc = shown["recorte"]
            bottom = overlay(bi, bm, bc, (CROP_W, CROP_H))
            cv2.imwrite(os.path.join(args.images_dir, "frame_{}.png".format(frame)), np.vstack([top, bottom]))
            saved += 1

    n = len(stats["global-RGB"]["hit"])
    print("\nDiagnóstico do WASB ({} frames com bola; position {})\n".format(n, args.position))
    print("{:<19}{:>10}{:>14}{:>12}".format("", "pico (med)", "na bola (med)", "acerto 25px"))
    for name in names:
        s = stats[name]
        print("{:<19}{:>10.2f}{:>14.2f}{:>11.0f}%".format(
            name, np.median(s["peak"]), np.median(s["near"]), 100.0 * np.mean(s["hit"])))
    if saved:
        print("\nimagens: {} (em cima o frame inteiro, embaixo o recorte; verde = bola real)".format(args.images_dir))


if __name__ == "__main__":
    main()