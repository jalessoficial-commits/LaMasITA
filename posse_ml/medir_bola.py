"""Mede o perfil do nosso tracker da bola e grava posse_ml/modelos/perfil_bola.json.

    python -m posse_ml.medir_bola
    python -m posse_ml.medir_bola --estados evaluation/results/primeira_rodada.csv

O perfil é o que a degradação da DFL imita (perfil.py). Para separar a bola viva da morta é preciso
um rótulo por frame: por padrão, o GT público (5 minutos). Com --estados, o jogo inteiro: o melhor é
o detector de paradas da equipe (analise/out/paradas_D.csv, frame,morta), que não depende deste
pacote. Uma predição do próprio posse_ml também serve, mas é circular: as lacunas longas que ele
confunde com parada seriam medidas como lacunas de bola morta.

A câmera tática sai da homografia, com o estimate_camera do posse/predictor.py.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from . import config, perfil
from .config import LETRAS, X


def ler_rotulos(caminho, n):
    """Rótulo por frame. Aceita frame,label (H/A/D/X) e frame,morta (1 = bola morta, como o
    analise/out/paradas_D.csv do detector de paradas); no segundo, bola viva vira H, porque aqui
    só importa separar viva de morta."""
    tab = pd.read_csv(caminho)
    if "label" not in tab.columns and "morta" in tab.columns:
        tab["label"] = np.where(tab["morta"].astype(float) >= 0.5, "D", "H")
    rot = np.full(n, X, dtype=np.int8)
    codigo = {letra: i for i, letra in enumerate(LETRAS)}
    f = tab["frame"].to_numpy(dtype=np.int64)
    ok = (f >= 0) & (f < n)
    rot[f[ok]] = [codigo.get(str(v), X) for v in tab["label"].to_numpy()[ok]]
    return rot


def camera_tatica(tempos, campo, n):
    """{tempo: [x, y, altura]} em metros, ou None se a homografia não puder ser lida."""
    sys.path.insert(0, os.path.join(config.RAIZ, "posse"))
    try:
        import predictor
        centros, resumo = predictor.estimate_camera(tempos, campo, n)
    except Exception as erro:  # sem homografia, a degradação sorteia a câmera
        print("câmera: não estimada ({})".format(erro))
        return None
    if centros is None:
        print("câmera: não estimada ({})".format(resumo))
        return None
    print("câmera:", resumo)
    return {str(t): [round(float(v), 2) for v in centros[a]] for t, a, _ in tempos}


def main(argv=None):
    from games import GAME, GAMES, half_video_blocks, source_frames
    from .partida import ler_bola_csv

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--bola", help="outro CSV da bola (em 0-1 ou pixels); padrão: a bola da instância")
    ap.add_argument("--estados", help="bola viva/morta: frame,label ou frame,morta. Padrão: "
                    "analise/out/paradas_D.csv (jogo inteiro) se existir, senão o GT público (5 min)")
    ap.add_argument("--sem-camera", action="store_true")
    ap.add_argument("--out", help="padrão: modelos/<instância>/perfil_bola.json")
    args = ap.parse_args(argv)
    args.bola = args.bola or config.bola_da_instancia(args.instancia)
    args.out = args.out or os.path.join(config.pasta_modelos(args.instancia), "perfil_bola.json")
    if args.estados is None and os.path.isfile(config.PARADAS_EQUIPE):
        args.estados = config.PARADAS_EQUIPE

    n = source_frames("tactical", GAME)
    campo = tuple(GAMES[GAME]["pitch"])
    tempos = half_video_blocks(GAME)
    if args.bola == "atd":
        from .partida import carregar_jogo
        caminho, bola = "bola do ATD", carregar_jogo(bola="atd").bola
    else:
        caminho = config.caminho_bola(args.bola)
        bola = ler_bola_csv(caminho, n) * np.array(campo, dtype=np.float32)
    rotulos = ler_rotulos(args.estados or config.GT_PUBLICO, n)
    camera = None if args.sem_camera else camera_tatica(tempos, campo, n)
    origem = "estados de {}".format(os.path.basename(args.estados)) if args.estados else "GT público"
    p = perfil.medir(bola, tempos, rotulos, camera,
                     descricao="{}; viva/morta pelo {}".format(os.path.basename(caminho), origem))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    perfil.salvar(p, args.out)
    f = p["falta"]
    print("bola presente em {:.1%} dos frames avaliados; pulos: {:.1%} das posições".format(
        p["presenca"], p["pulos"]["taxa"]))
    print("falta com a bola viva {:.1%} ({} frames), com a bola morta {:.1%} ({} frames)".format(
        f["viva"] or 0, f["frames_viva"], f["morta"] or 0, f["frames_morta"]))
    for k in ("viva", "morta"):
        if p["lacunas"][k] is None:
            print("lacunas de bola {}: só {}, uso o histograma de todas as lacunas".format(
                k, p["lacunas"]["n_" + k]))
    print("gravado em", args.out)


if __name__ == "__main__":
    main()
