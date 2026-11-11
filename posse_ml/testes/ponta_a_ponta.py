"""Roda o pacote inteiro com dados sintéticos numa pasta temporária: medir a bola, treinar,
calibrar e gerar a submissão. Serve para conferir a instalação sem os dados do desafio nem a DFL.

    python -m posse_ml.testes.ponta_a_ponta            # ~5 min
    python -m posse_ml.testes.ponta_a_ponta --manter   # não apaga a pasta no fim
    python -m posse_ml.testes.ponta_a_ponta --classificadores xgboost gru   # + o comparar_classificadores

Nada do que sai daqui vale para o jogo de verdade: os números só mostram que tudo encaixa.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from posse_ml import config, degradacao, metrica
from posse_ml.config import D, FPS, X
from posse_ml.janelas import trechos
from posse_ml.testes import sintetico

DESLOCAMENTOS = {1: 11000, 2: 28000}       # tático = transmissão + d, inventados
CAMERA = (52.65, 68.0 + 35.0, 22.0)


def montar(raiz):
    from games import GAMES, half_video_blocks, source_frames
    jogo = "BAR-ATH"
    campo = tuple(GAMES[jogo]["pitch"])
    n_tat = source_frames("tactical", jogo)
    n_tra = source_frames("broadcast", jogo)
    layout = (n_tat, half_video_blocks(jogo), config.KICKOFFS[jogo])
    print("gerando o jogo sintético ({} frames)...".format(n_tat), flush=True)
    p = sintetico.jogo(semente=11, layout=layout, comp=campo[0], larg=campo[1])
    # a "bola do tracker": a bola sintética com lacunas e pulos, como a degradação faz na DFL
    q, _ = degradacao.degradar(p, np.random.default_rng(5))

    tat = os.path.join(raiz, "data", "tactical")
    tra = os.path.join(raiz, "data", "broadcast")
    os.makedirs(tat)
    os.makedirs(tra)
    sintetico.escrever_atd(p, os.path.join(tat, jogo + "_tactical_atd.xml"), os.path.join(tat, jogo + "_tactical_atd.txt"))
    sintetico.escrever_bola_csv(q.bola, campo, os.path.join(tat, jogo + "_tactical_ball_tracker.csv"))
    hom = sintetico.homografia(CAMERA, campo)
    sintetico.escrever_homografia(os.path.join(tat, jogo + "_tactical_homography.csv"), n_tat, hom)
    # a mesma bola em pixels do vídeo, só nos frames com bola, como o tracking_bola/out/viterbi_bola.csv
    norm = q.bola / np.array(campo, dtype=np.float32)
    with open(os.path.join(raiz, "bola_px.csv"), "w") as arq:
        arq.write("frame,x,y,regra\n")
        for f in np.flatnonzero(np.isfinite(norm[:, 0])):
            u, v, w = hom @ np.r_[norm[f], 1.0]
            arq.write("{},{:.2f},{:.2f},detector\n".format(f, u / w, v / w))

    # transmissão: cada tempo no seu deslocamento; na maioria das paradas longas, corta para close
    rng = np.random.default_rng(3)
    jb = np.full((n_tra, 22, 2), np.nan, dtype=np.float32)
    bb = np.full((n_tra, 2), np.nan, dtype=np.float32)
    valida = np.zeros(n_tra, dtype=bool)
    for tempo, a, b in p.tempos:
        d = DESLOCAMENTOS[tempo]
        f = np.arange(max(a - d, 0), min(b - d + 1, n_tra))
        jb[f] = p.jogadores[f + d]
        bb[f] = p.bola[f + d] + rng.normal(0, 0.3, (len(f), 2)).astype(np.float32)
        valida[f] = True
        morta = p.rotulos[f + d] == D
        for s, e in trechos(morta):
            if e - s > 6 * FPS and rng.random() < 0.7:
                corte = slice(f[s] + 2 * FPS, f[e - 1] - FPS)
                jb[corte], bb[corte], valida[corte] = np.nan, np.nan, False
    pb = p.trocar(jogadores=jb, bola=bb, lado=p.lado[:n_tra])
    sintetico.escrever_atd(pb, os.path.join(tra, jogo + "_broadcast_atd.xml"), os.path.join(tra, jogo + "_broadcast_atd.txt"))
    sintetico.escrever_homografia(os.path.join(tra, jogo + "_broadcast_homography.csv"), n_tra,
                                  sintetico.homografia(CAMERA, campo), valida)

    # GT "público": os primeiros 5 minutos do 1º tempo
    gt = np.full(n_tat, X, dtype=np.int8)
    ko = config.KICKOFFS[jogo][1]
    gt[ko:ko + 5 * 60 * FPS] = p.rotulos[ko:ko + 5 * 60 * FPS]
    caminho_gt = os.path.join(raiz, "gt_publico.csv")
    with open(caminho_gt, "w") as arq:
        arq.write("frame,label\n")
        arq.writelines("{},{}\n".format(i, config.LETRAS[c]) for i, c in enumerate(gt))

    # um "detector de paradas" de mentira: as paradas certas com as bordas 1 s atrasadas
    morta = p.rotulos == D
    nuno = np.zeros(n_tat, dtype=bool)
    for s, e in trechos(morta):
        nuno[min(s + FPS, e):min(e + FPS, n_tat)] = True
    with open(os.path.join(raiz, "paradas_D.csv"), "w") as arq:
        arq.write("frame,morta\n")
        arq.writelines("{},{}\n".format(i, int(m)) for i, m in enumerate(nuno))

    # três "jogos da DFL"
    pasta_dfl = os.path.join(raiz, "dfl")
    os.makedirs(pasta_dfl)
    for k in range(3):
        sintetico.escrever_dfl(sintetico.jogo(900, semente=100 + k), pasta_dfl, "SINT{}".format(k))
    return caminho_gt, pasta_dfl


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manter", action="store_true")
    ap.add_argument("--classificadores", nargs="*", default=[],
                    help="roda também o comparar_classificadores com estes (hgb, xgboost, lightgbm, catboost, "
                         "gru, lstm, tcn, transformer), em versão pequena")
    args = ap.parse_args(argv)
    raiz = tempfile.mkdtemp(prefix="posse_ml_")
    t0 = time.time()
    try:
        gt, pasta_dfl = montar(raiz)
        # a bola padrão (o rastro do Viterbi, em pixels) vem do bola_px.csv
        env = dict(os.environ, MTAH_DATA_DIR=os.path.join(raiz, "data"), POSSE_ML_GT=gt,
                   POSSE_ML_DFL=pasta_dfl, POSSE_ML_CACHE=os.path.join(raiz, "cache"),
                   POSSE_ML_BOLA=os.path.join(raiz, "bola_px.csv"), POSSE_ML_PARADAS=os.path.join(raiz, "paradas_D.csv"),
                   POSSE_ML_RESULTADOS=os.path.join(raiz, "resultados"),
                   POSSE_ML_MODELOS=os.path.join(raiz, "modelos"))
        mod = os.path.join(raiz, "modelos", "viterbi")
        passos = [
            ["posse_ml.medir_bola", "--out", os.path.join(mod, "perfil_bola.json")],
            ["posse_ml.treinar", "--perfil", os.path.join(mod, "perfil_bola.json"), "--modelos", mod,
             "--hiper", "max_iter=40,min_samples_leaf=50", "--passo", "3", "--processos", "2"],
            ["posse_ml.calibrar", "--modelos", mod],
            ["posse_ml", "--modelos", mod, "--out", os.path.join(raiz, "submissao.csv")],
            ["posse_ml", "--modelos", mod, "--out", os.path.join(raiz, "submissao_fusao.csv"),
             "--instancia", "fusao"],
            ["posse_ml.comparar_bolas"],
            ["posse_ml", "--modelos", mod],
            ["posse_ml.comparar_paradas"],
            ["posse_ml", "--modelos", mod, "--paradas", gt],
            ["posse_ml", "--modelos", mod, "--paradas", os.path.join(raiz, "paradas_D.csv")],
            ["posse_ml.calibrar", "--modelos", mod, "--paradas", os.path.join(raiz, "paradas_D.csv")],
            ["posse_ml", "--modelos", mod, "--paradas", os.path.join(raiz, "paradas_D.csv"), "--modo-paradas", "suave"],
            ["posse_ml.testar_arvores", "--teste", "--arvores", "50", "400", "--folhas", "15"],
        ]
        if args.classificadores:
            # os outros pelo comparar_classificadores, com modelos pequenos; o hgb já está treinado acima
            passos.append(["posse_ml.comparar_classificadores", "--teste", "--classificadores", "hgb"]
                          + [c for c in args.classificadores if c != "hgb"])
            outros = [c for c in args.classificadores if c != "hgb"]
            if outros:
                passos.append(["posse_ml", "--classificador", outros[0]])
            if len(outros) >= 2:
                combo = os.path.join(mod, "+".join(outros[:2]))
                passos += [["posse_ml.combinar", "--classificadores"] + outros[:2],
                           ["posse_ml", "--modelos", combo, "--out", os.path.join(raiz, "sub_combo.csv")]]
            if outros:
                passos.append(["posse_ml.testar_sementes", "--teste", "--classificadores", outros[0],
                               "--sementes", "1"])
            if "tcn" in outros:
                passos.append(["posse_ml.testar_variantes", "--teste", "--variantes", "lstm", "tcn_e20", "transformer", "tcn_e5"])
                passos.append(["posse_ml.testar_epocas", "--teste", "--classificadores", "tcn", "--epocas", "5", "10",
                               "--sementes", "0", "1"])
                passos.append(["posse_ml.testar_epocas", "--teste", "--classificadores", "tcn", "--epocas", "5", "10",
                               "--sementes", "0", "1", "--validacao", "igual"])
                if metrica.jogos():
                    passos.append(["posse_ml.testar_dados", "--teste", "--classificadores", "tcn", "--sementes", "7"])
                passos.append(["posse_ml.testar_configs", "--teste", "--configs", "tcn_d03", "tcn_b4", "--epocas", "5",
                               "--sementes", "0", "1"])
        for passo in passos:
            print("\n$ python -m " + " ".join(passo), flush=True)
            r = subprocess.run([sys.executable, "-m"] + passo, cwd=config.RAIZ, env=env)
            if r.returncode:
                sys.exit("falhou: {}".format(passo[0]))
        a = open(os.path.join(raiz, "submissao.csv")).read().split()
        b = open(os.path.join(raiz, "submissao_fusao.csv")).read().split()
        iguais = np.mean([x == y for x, y in zip(a, b)])
        print("\nbola em pixels x bola em 0-1: {:.3%} dos frames com o mesmo rótulo".format(iguais))
        if iguais < 0.99:
            sys.exit("a bola em pixels deu outro resultado")
        print("tudo certo em {:.0f} s ({})".format(time.time() - t0, raiz))
    finally:
        if not args.manter:
            shutil.rmtree(raiz, ignore_errors=True)


if __name__ == "__main__":
    main()
