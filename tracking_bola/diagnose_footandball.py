"""Por que o FootAndBall acha tão pouco no seu vídeo? Testa escala e cor, e mostra o que a rede vê.

Colocar em tracking_bola/, junto de detect_footandball.py (este script importa dele):

    python tracking_bola/diagnose_footandball.py --fb-dir tracking_bola/third_party/FootAndBall
    python tracking_bola/diagnose_footandball.py --fb-dir tracking_bola/third_party/FootAndBall --scales 1 1.5 2 --frames 40

O que se sabe (artigo e código do FootAndBall): a rede é totalmente convolucional e foi treinada em frames
Full-HD do ISSIA-CNR, o MESMO tamanho do seu vídeo (1920x1080), com aumento de escala aleatório entre 0,8 e
1,2. Então o tamanho da tela em si não é o problema. O que pode ser: a bola do seu vídeo (~8 px) ser menor
que a do treino (o artigo fala em 8 a 20 px), a imagem ter outro aspecto (câmera tática mais aberta,
compressão, cor), ou a ordem de cores trocada. Este script separa essas hipóteses.

Para cada escala pedida (e uma condição extra, escala 1.0 com canais BGR), em N frames em que você marcou a bola,
roda a rede UMA vez e olha o mapa de confiança da bola (o que a rede produz antes de qualquer limiar):

  conf na bola     maior confiança do mapa a até --radius px (original) do seu clique. A rede "vê" a bola
                   se isto for alto. Mediana sobre os frames.
  >= 0.2           % dos frames em que essa confiança passa de 0,2 (o limiar do detect_footandball.py).
  bola é o pico    % dos frames em que o maior pico do mapa inteiro está na bola (ela venceu tudo que há
                   fora do raio). Se a conf na bola é baixa mas o pico é alto, a rede acha OUTRA coisa.
  pico falso       mediana da confiança do maior pico fora da bola.

E, por condição, os 3 lugares onde mais cai o maior pico quando ele NÃO está na bola (em blocos de 40 px do
vídeo original). Se um lugar se repete em muitos frames, é um objeto fixo (marca do campo, placar, logo):
a rede está confundindo isso com a bola. Se os lugares são espalhados, é confusão com jogadores/ruído.

Como ler: se alguma escala maior (1.5, 2.0) levanta bastante a "conf na bola", o problema é o tamanho da
bola, e dá para rodar o FootAndBall nessa escala (custa mais tempo: 2x = 4x). Se NENHUMA escala levanta, a
rede não generaliza para esse vídeo e não vale insistir. Se só o BGR melhora, o vídeo está com canais trocados.
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
import detect_footandball as dfb  # noqa: E402

DEFAULT_ANNOTATIONS = dfb.DEFAULT_ANNOTATIONS
CELL = 4          # a rede da bola devolve 1 célula por bloco de 4x4 px
BLOCK = 40        # tamanho dos blocos do agrupamento dos picos falsos (px)


def load_map_model(args):
    """(função imagem BGR -> mapa de confiança da bola [h, w], device)."""
    fb_dir = os.path.abspath(args.fb_dir)
    if not os.path.isfile(os.path.join(fb_dir, "network", "footandball.py")):
        sys.exit("não achei o FootAndBall em {}".format(fb_dir))
    weights = os.path.abspath(args.weights or os.path.join(fb_dir, dfb.DEFAULT_WEIGHTS))
    if not os.path.isfile(weights):
        sys.exit("não achei os pesos em {}. Use --weights.".format(weights))
    sys.path.insert(0, fb_dir)
    import torch
    import network.footandball as footandball
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # o construtor só aceita 'train', 'test' ou 'detect'; a fase 'eval' (mapas já com softmax) se ajusta depois
    model = footandball.model_factory("fb1", "detect")
    model.phase = "eval"
    try:
        state = torch.load(weights, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
    model = model.to(device)
    model.eval()

    def ball_map(image_bgr, bgr):
        tensor = torch.from_numpy(dfb.to_tensor_array(image_bgr, bgr)).to(device)
        with torch.no_grad():
            _, _, ball = model(tensor)             # (1, h, w, 2), já passou pelo softmax
        return ball[0, :, :, 1].cpu().numpy()

    return ball_map, device


def cell_centers_original(shape, sx, sy):
    """Coordenadas (x, y) no vídeo original do centro de cada célula do mapa (h, w)."""
    h, w = shape
    xs = (CELL * np.arange(w) + (CELL - 1) / 2.0 + 0.5) / sx - 0.5
    ys = (CELL * np.arange(h) + (CELL - 1) / 2.0 + 0.5) / sy - 0.5
    return xs, ys


def analyse(prob, truth, sx, sy, radius):
    """(conf perto da bola, conf do maior pico fora do raio, (x, y) desse pico, a bola é o maior pico?)."""
    xs, ys = cell_centers_original(prob.shape, sx, sy)
    near = (np.hypot(xs[None, :] - truth[0], ys[:, None] - truth[1]) <= radius)
    if not near.any():                              # raio menor que o espaçamento das células: pega a mais próxima
        distance = np.hypot(xs[None, :] - truth[0], ys[:, None] - truth[1])
        near = distance == distance.min()
    conf_near = float(prob[near].max())
    outside = np.where(near, -1.0, prob)
    index = np.unravel_index(int(np.argmax(outside)), outside.shape)
    far_conf = float(outside[index])
    return conf_near, far_conf, (float(xs[index[1]]), float(ys[index[0]])), conf_near >= far_conf


def summarize(results, radius):
    near = np.array([r[0] for r in results])
    far = np.array([r[1] for r in results])
    wins = np.array([r[3] for r in results])
    clusters = {}
    for r in results:
        if not r[3]:
            key = (int(r[2][0] // BLOCK) * BLOCK, int(r[2][1] // BLOCK) * BLOCK)
            clusters[key] = clusters.get(key, 0) + 1
    top = sorted(clusters.items(), key=lambda kv: -kv[1])[:3]
    return {"n": len(results), "near_median": float(np.median(near)), "ge02": 100.0 * np.mean(near >= 0.2),
            "wins": 100.0 * np.mean(wins), "far_median": float(np.median(far)), "top": top}


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fb-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.75, 1.0, 1.25, 1.5, 2.0],
                        help="escalas a testar (padrão 0.75 1 1.25 1.5 2)")
    parser.add_argument("--frames", type=int, default=60, help="quantos frames com bola usar (padrão 60, espalhados)")
    parser.add_argument("--radius", type=float, default=12.0, help="raio (px do original) que conta como 'na bola' (padrão 12)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args(argv)
    if model_factory is None and not args.fb_dir:
        parser.error("--fb-dir é obrigatório")
    import cv2
    if not os.path.isfile(args.annotations):
        sys.exit("não achei {}".format(args.annotations))
    annotations = read_annotations(args.annotations)
    n_frames = source_frames("tactical", GAME)
    balls = sorted((f, (v[1], v[2])) for f, v in annotations.items() if v[0] == "ball" and 0 <= f < n_frames)
    if not balls:
        sys.exit("nenhum frame com bola anotada")
    if len(balls) > args.frames:
        picks = np.linspace(0, len(balls) - 1, args.frames).round().astype(int)
        balls = [balls[i] for i in picks]
    ball_map, device = (model_factory or load_map_model)(args)
    print("dispositivo {}, {} frames com bola, raio {} px".format(device, len(balls), args.radius))

    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")
    conditions = [("escala {:g} RGB".format(s), s, False) for s in args.scales] + [("escala 1 BGR", 1.0, True)]
    results = {name: [] for name, _, _ in conditions}
    for count, (frame, truth) in enumerate(balls, start=1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            continue
        for name, scale, bgr in conditions:
            small, sx, sy = dfb.resize_for_net(image, scale)
            results[name].append(analyse(ball_map(small, bgr), truth, sx, sy, args.radius))
        if count % 10 == 0:
            print("  {}/{} frames".format(count, len(balls)))

    print("\n{:<16} {:>3} {:>14} {:>8} {:>14} {:>16}".format(
        "condição", "n", "conf na bola", ">= 0.2", "bola é o pico", "pico falso (med)"))
    summaries = {}
    for name, _, _ in conditions:
        s = summaries[name] = summarize(results[name], args.radius)
        print("{:<16} {:>3} {:>14.3f} {:>7.1f}% {:>13.1f}% {:>16.3f}".format(
            name, s["n"], s["near_median"], s["ge02"], s["wins"], s["far_median"]))
    print("\nOnde cai o maior pico quando NÃO é a bola (bloco de {} px do vídeo original: x, y -> frames):".format(BLOCK))
    for name, _, _ in conditions:
        print("  {:<16} {}".format(name, "   ".join("({}, {}) -> {}".format(k[0], k[1], c) for k, c in summaries[name]["top"]) or "-"))
    print("\nLeitura: 'conf na bola' alta em alguma escala = problema de tamanho; baixa em todas = a rede não generaliza")
    print("para este vídeo. Um mesmo bloco repetido nos picos falsos = objeto fixo sendo confundido com a bola.")


if __name__ == "__main__":
    main()