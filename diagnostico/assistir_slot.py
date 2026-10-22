"""Assiste um trecho do vídeo tactical com o tracking (ATD) desenhado por cima.

    python diagnostico/assistir_slot.py 24                 # começa onde o slot 24 aparece pela 1ª vez
    python diagnostico/assistir_slot.py 24 30000           # 1º frame >= 30000 com o slot 24
    python diagnostico/assistir_slot.py 24 30000 60        # idem, por 60 segundos
    python diagnostico/assistir_slot.py 24 30000 60 --salvar   # grava um .mp4 em vez de abrir janela
    python diagnostico/assistir_slot.py 24 30000 10 --exato    # começa exatamente no frame 30000

Cores: casa vermelho, fora azul, bola branca, slot escolhido amarelo e maior. Todo slot leva o
número ao lado; o topo mostra quantos slots de cada time estão presentes no frame.
Teclas na janela: espaço pausa/continua, q sai.

Precisa de: pip install opencv-python. Usa o cache do diagnostico.py
(data/_cache/atd_xy_v2.npz), então rode o diagnóstico antes. O .mp4 salvo vai para data/_cache/,
que o git ignora: é vídeo recortado da partida, não pode ir para o repositório público.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))  # baseline, load_homography
from games import DATA_DIR, asset_paths  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
import baseline  # noqa: E402

# Params (cores em BGR, como o OpenCV usa)
CACHE = os.path.join(DATA_DIR, "_cache", "atd_xy_v2.npz")
FPS = 25
COR_CASA, COR_FORA, COR_BOLA, COR_SLOT = (0, 0, 255), (255, 0, 0), (255, 255, 255), (0, 255, 255)


def na_tela(H, pos, frames, largura=1920, altura=1080):
    """Máscara sobre `frames`: a posição (normalizada) projeta dentro da imagem?"""
    M, x, y = H[frames], pos[frames, 0], pos[frames, 1]
    den = M[:, 2, 0] * x + M[:, 2, 1] * y + M[:, 2, 2]
    px = (M[:, 0, 0] * x + M[:, 0, 1] * y + M[:, 0, 2]) / den
    py = (M[:, 1, 0] * x + M[:, 1, 1] * y + M[:, 1, 2]) / den
    return (px >= 0) & (px < largura) & (py >= 0) & (py < altura)


def desenhar(img, f, xy, ball, team, H, slot):
    for s in range(xy.shape[1]):
        if np.isnan(xy[f, s, 0]):
            continue
        px, py = project(H[f], xy[f, s, 0], xy[f, s, 1])
        if s == slot:
            cv2.circle(img, (int(px), int(py)), 18, COR_SLOT, 4)
            cv2.putText(img, str(slot + 1), (int(px) + 20, int(py) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COR_SLOT, 2)
        else:
            cor = COR_CASA if team[s] == 0 else COR_FORA
            cv2.circle(img, (int(px), int(py)), 9, cor, 3)
            cv2.putText(img, str(s + 1), (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    if not np.isnan(ball[f, 0]):
        px, py = project(H[f], ball[f, 0], ball[f, 1])
        cv2.circle(img, (int(px), int(py)), 7, COR_BOLA, -1)
    if np.isnan(xy[f, slot, 0]):
        estado = "ausente"
    else:
        estado = "na tela" if na_tela(H, xy[:, slot], np.array([f]))[0] else "fora da tela"
    vivos = ~np.isnan(xy[f, :, 0])
    status = "slot %d %s | bola %s | casa %d fora %d" % (
        slot + 1, estado, "ausente" if np.isnan(ball[f, 0]) else "presente",
        np.sum(vivos & (team == 0)), np.sum(vivos & (team == 1)))
    cv2.putText(img, "frame %d  (%.1f s)  %s" % (f, f / FPS, status), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(img, "frame %d  (%.1f s)  %s" % (f, f / FPS, status), (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slot", type=int, help="número do slot (1 a 46)")
    parser.add_argument("inicio", type=int, nargs="?",
                        help="começa no 1º frame >= este em que o slot aparece (padrão: 0)")
    parser.add_argument("segundos", type=float, nargs="?", default=30.0, help="duração (padrão 30)")
    parser.add_argument("--salvar", action="store_true", help="grava .mp4 em vez de abrir janela")
    parser.add_argument("--exato", action="store_true",
                        help="começa exatamente no frame `inicio`, sem procurar o slot")
    args = parser.parse_args()

    cached = np.load(CACHE)
    xy, ball = cached["xy"], cached["ball"]
    team = baseline.team_of_each_track(asset_paths()["atd_meta"])
    H, _, valid = load_homography()
    slot = args.slot - 1

    presente = np.flatnonzero(~np.isnan(xy[:, slot, 0]) & valid)
    visivel = presente[na_tela(H, xy[:, slot], presente)]
    print("slot %d presente em %d frames, dentro da imagem em %d" % (
        args.slot, len(presente), len(visivel)))
    # Começa no primeiro frame, a partir de `inicio`, em que o slot aparece NA IMAGEM: presente
    # no tracking não basta, porque a câmera não mostra o campo inteiro.
    depois = visivel[visivel >= (args.inicio or 0)]
    if args.exato:
        inicio = args.inicio or 0
    elif not len(depois):
        print("slot %d não aparece na imagem a partir do frame %d" % (args.slot, args.inicio or 0))
        return
    else:
        inicio = int(depois[0])
    print("começando no frame %d" % inicio)
    fim = min(inicio + int(args.segundos * FPS), len(xy))

    video = cv2.VideoCapture(asset_paths()["video"])
    video.set(cv2.CAP_PROP_POS_FRAMES, inicio)   # um único seek; depois a leitura é sequencial

    writer = None
    if args.salvar:
        out = os.path.join(DATA_DIR, "_cache", "slot%d_%d_%d.mp4" % (args.slot, inicio, fim))
        os.makedirs(os.path.dirname(out), exist_ok=True)
    else:
        cv2.namedWindow("ATD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ATD", 1280, 720)

    pausado = False
    f = inicio
    while f < fim:
        if not pausado:
            ok, img = video.read()
            if not ok:
                print("o vídeo acabou ou falhou no frame %d" % f)
                break
            if valid[f]:
                img = desenhar(img, f, xy, ball, team, H, slot)
            if writer is None and args.salvar:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            if args.salvar:
                writer.write(img)
            else:
                cv2.imshow("ATD", img)
            f += 1
        if not args.salvar:
            key = cv2.waitKey(1000 // FPS) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                pausado = not pausado

    video.release()
    if writer is not None:
        writer.release()
        print("salvo:", out)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()