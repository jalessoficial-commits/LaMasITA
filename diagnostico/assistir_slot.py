"""Assiste um trecho do vídeo tactical com o tracking (ATD) desenhado por cima.

Uso
---
    python diagnostico/assistir_slot.py --inicio 30000                    # 30 s a partir do frame 30000
    python diagnostico/assistir_slot.py --slot 24                         # 1ª vez que o slot 24 aparece na imagem
    python diagnostico/assistir_slot.py --slot 24 --inicio 30000          # 1ª vez a partir do frame 30000
    python diagnostico/assistir_slot.py --slot 24 --inicio 30000 --exato  # exatamente no frame 30000
    python diagnostico/assistir_slot.py --slot 24 --segundos 60 --salvar  # grava .mp4 em vez de abrir janela

Argumentos
----------
    --slot N       slot a destacar (1 a 46). Opcional: sem ele, nada é destacado.
    --inicio F     frame de vídeo (0 a 199318) a partir do qual começar. Padrão: 0.
    --segundos S   duração do trecho. Padrão: 30.
    --exato        começa exatamente em --inicio, sem procurar o slot.
    --salvar       grava um .mp4 em data/_cache/ em vez de abrir uma janela.

Onde o trecho começa
--------------------
    sem --slot                 em --inicio
    com --slot, sem --exato    no 1º frame >= --inicio em que o slot está presente no tracking
                               E projeta dentro da imagem. Presente não basta: a câmera não
                               mostra o campo inteiro, e o ATD guarda posições fora dela.
    com --slot e --exato       em --inicio

O trecho dura --segundos a partir daí e continua mesmo que o slot suma ou saia da imagem.

O que aparece na tela
---------------------
    círculo vermelho + número    slot da casa (Barcelona)
    círculo azul + número        slot de fora (Athletic)
    ponto branco                 bola, quando o ATD tem a bola
    círculo amarelo maior        o slot escolhido em --slot
    topo                         frame, tempo de vídeo, estado do slot (na tela / fora da tela /
                                 ausente), bola presente ou ausente, slots presentes por time

Teclas (com a janela do vídeo selecionada)
------------------------------------------
    espaço    pausa / continua
    d         com o vídeo pausado, avança um frame
    q         sai (fechar a janela pelo X também sai; Ctrl+C no terminal, em último caso)

Requisitos e cuidados
---------------------
    pip install opencv-python. Usa o cache do diagnostico.py (data/_cache/atd_xy_v2.npz):
    rode o diagnóstico antes.

    A posição de um slot fora da imagem pode ser estimada pelo tracker, não observada: o goleiro
    do Barcelona (slot 24) deixa um "fantasma" parado quando o slot 46 passa a segui-lo.

    O .mp4 gravado é vídeo recortado da partida. Fica em data/_cache/, que o git ignora, e não
    pode ir para o repositório público.

    Se os círculos não caírem em cima dos jogadores, o vídeo e o tracking estão dessincronizados
    (a busca de frame do OpenCV pode errar em alguns codecs). O script avisa quando a posição
    reportada pelo vídeo difere da pedida.
"""

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "starter"))  # baseline, load_homography
from games import DATA_DIR, GAME, GAMES, asset_paths  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
import baseline  # noqa: E402

# Params
CACHE = os.path.join(DATA_DIR, "_cache", "atd_xy_v2.npz")
OUT_DIR = os.path.join(DATA_DIR, "_cache")
FPS = GAMES[GAME]["fps"]
LARGURA, ALTURA = 1920, 1080                       # resolução do vídeo tactical (DATA.md)
JANELA = "ATD"
# Cores em BGR, a ordem que o OpenCV usa.
COR_CASA, COR_FORA = (0, 0, 255), (255, 0, 0)
COR_BOLA, COR_SLOT = (255, 255, 255), (0, 255, 255)


def carregar():
    """(xy, ball, team, H, valid) para o vídeo tactical inteiro.

    xy (frames, slots, 2) e ball (frames, 2) em coordenadas normalizadas 0-1, NaN = ausente.
    team[s] é 0 (casa) ou 1 (fora). H (frames, 3, 3) leva normalizado -> pixel; valid diz se a
    matriz existe naquele frame.
    """
    if not os.path.isfile(CACHE):
        sys.exit("cache não encontrado: %s\nRode antes: python diagnostico/diagnostico.py"
                 % CACHE)
    cached = np.load(CACHE)
    team = baseline.team_of_each_track(asset_paths()["atd_meta"])
    H, _, valid = load_homography()
    return cached["xy"], cached["ball"], team, H, valid


def na_tela(H, pos, frames):
    """Máscara sobre `frames`: a posição normalizada `pos[frame]` projeta dentro da imagem?"""
    M, x, y = H[frames], pos[frames, 0], pos[frames, 1]
    den = M[:, 2, 0] * x + M[:, 2, 1] * y + M[:, 2, 2]
    px = (M[:, 0, 0] * x + M[:, 0, 1] * y + M[:, 0, 2]) / den
    py = (M[:, 1, 0] * x + M[:, 1, 1] * y + M[:, 1, 2]) / den
    return (px >= 0) & (px < LARGURA) & (py >= 0) & (py < ALTURA)


def escolher_inicio(slot, inicio, exato, xy, H, valid):
    """Frame em que o trecho começa (ver "Onde o trecho começa" no topo), ou None se não há."""
    if slot is None or exato:
        return inicio
    candidatos = np.flatnonzero(~np.isnan(xy[:, slot, 0]) & valid)
    candidatos = candidatos[candidatos >= inicio]
    visiveis = candidatos[na_tela(H, xy[:, slot], candidatos)]
    print("slot %d a partir do frame %d: presente em %d frames, na imagem em %d" % (
        slot + 1, inicio, len(candidatos), len(visiveis)))
    return int(visiveis[0]) if len(visiveis) else None


def texto(img, linha, y):
    """Texto branco com contorno preto, legível sobre qualquer fundo."""
    for cor, espessura in (((0, 0, 0), 6), ((255, 255, 255), 2)):
        cv2.putText(img, linha, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, espessura)


def desenhar(img, f, xy, ball, team, H, slot):
    """Desenha os slots, a bola e a linha de status do frame `f` sobre `img`."""
    for s in np.flatnonzero(~np.isnan(xy[f, :, 0])):
        px, py = (int(v) for v in project(H[f], xy[f, s, 0], xy[f, s, 1]))
        if s == slot:
            cv2.circle(img, (px, py), 18, COR_SLOT, 4)
            cv2.putText(img, str(s + 1), (px + 20, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COR_SLOT, 2)
        else:
            cor = COR_CASA if team[s] == 0 else COR_FORA
            cv2.circle(img, (px, py), 9, cor, 3)
            cv2.putText(img, str(s + 1), (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    if not np.isnan(ball[f, 0]):
        px, py = (int(v) for v in project(H[f], ball[f, 0], ball[f, 1]))
        cv2.circle(img, (px, py), 7, COR_BOLA, -1)

    vivos = ~np.isnan(xy[f, :, 0])
    partes = ["frame %d (%.1f s)" % (f, f / FPS)]
    if slot is not None:
        if not vivos[slot]:
            estado = "ausente"
        else:
            estado = "na tela" if na_tela(H, xy[:, slot], np.array([f]))[0] else "fora da tela"
        partes.append("slot %d %s" % (slot + 1, estado))
    partes.append("bola %s" % ("ausente" if np.isnan(ball[f, 0]) else "presente"))
    partes.append("casa %d fora %d" % (np.sum(vivos & (team == 0)), np.sum(vivos & (team == 1))))
    texto(img, " | ".join(partes), 50)
    return img


def reproduzir(inicio, fim, desenhar_frame, valid, saida=None):
    """Toca (ou grava em `saida`) os frames [inicio, fim) com `desenhar_frame(img, f)` aplicado."""
    video = cv2.VideoCapture(asset_paths()["video"])
    if not video.isOpened():
        sys.exit("não consegui abrir o vídeo: %s" % asset_paths()["video"])
    video.set(cv2.CAP_PROP_POS_FRAMES, inicio)     # uma única busca; depois a leitura é sequencial
    reportado = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    if reportado != inicio:
        print("AVISO: pedi o frame %d e o vídeo diz estar no %d; o desenho pode estar fora de "
              "sincronia" % (inicio, reportado))

    writer = None
    if saida is None:
        cv2.namedWindow(JANELA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(JANELA, 1280, 720)

    f, pausado, avancar = inicio, False, False
    while f < fim:
        if not pausado or avancar:
            ok, img = video.read()
            if not ok:
                print("o vídeo acabou ou falhou no frame %d" % f)
                break
            if valid[f]:
                img = desenhar_frame(img, f)
            if saida is not None:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
                writer.write(img)
            else:
                cv2.imshow(JANELA, img)
            f += 1
            avancar = False
        if saida is None:
            tecla = cv2.waitKey(1000 // FPS) & 0xFF
            if tecla == ord("q") or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla == ord(" "):
                pausado = not pausado
            elif tecla == ord("d") and pausado:
                avancar = True

    video.release()
    if writer is not None:
        writer.release()
        print("salvo:", saida)
    cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Detalhes completos no topo deste arquivo.")
    parser.add_argument("--slot", type=int, help="slot a destacar, 1 a 46 (opcional)")
    parser.add_argument("--inicio", type=int, default=0, help="frame de vídeo inicial (padrão 0)")
    parser.add_argument("--segundos", type=float, default=30.0, help="duração (padrão 30)")
    parser.add_argument("--exato", action="store_true",
                        help="começa exatamente em --inicio, sem procurar o slot")
    parser.add_argument("--salvar", action="store_true",
                        help="grava .mp4 em data/_cache/ em vez de abrir janela")
    args = parser.parse_args(argv)

    xy, ball, team, H, valid = carregar()
    n_frames, n_slots = xy.shape[:2]
    if args.slot is not None and not 1 <= args.slot <= n_slots:
        parser.error("--slot deve estar entre 1 e %d" % n_slots)
    if not 0 <= args.inicio < n_frames:
        parser.error("--inicio deve estar entre 0 e %d" % (n_frames - 1))
    if args.segundos <= 0:
        parser.error("--segundos deve ser positivo")
    slot = None if args.slot is None else args.slot - 1

    inicio = escolher_inicio(slot, args.inicio, args.exato, xy, H, valid)
    if inicio is None:
        print("o slot %d não aparece na imagem a partir do frame %d" % (args.slot, args.inicio))
        return
    fim = min(inicio + int(round(args.segundos * FPS)), n_frames)
    print("tocando os frames %d a %d (%.1f s)" % (inicio, fim - 1, (fim - inicio) / FPS))

    saida = None
    if args.salvar:
        os.makedirs(OUT_DIR, exist_ok=True)
        nome = "slot%d" % args.slot if args.slot is not None else "trecho"
        saida = os.path.join(OUT_DIR, "%s_%d_%d.mp4" % (nome, inicio, fim - 1))

    reproduzir(inicio, fim, lambda img, f: desenhar(img, f, xy, ball, team, H, slot), valid, saida)


if __name__ == "__main__":
    main()
