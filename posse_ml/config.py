"""Caminhos, códigos dos rótulos e os poucos números que valem para o pacote inteiro."""

import os
import sys

PASTA = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(PASTA)
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

# Tudo que é derivado dos dados do jogo (arrays em cache, probabilidades por frame) fica aqui, fora
# do Git (posse_ml/.gitignore). Os modelos treinados ficam em modelos/ e vão para o repositório.
CACHE = os.environ.get("POSSE_ML_CACHE") or os.path.join(PASTA, "cache")
MODELOS_BASE = os.environ.get("POSSE_ML_MODELOS") or os.path.join(PASTA, "modelos")
RESULTADOS = os.environ.get("POSSE_ML_RESULTADOS") or os.path.join(RAIZ, "evaluation", "results")
PARADAS_EQUIPE = os.environ.get("POSSE_ML_PARADAS") or os.path.join(RAIZ, "analise", "out", "paradas_D.csv")

# Uma instância por bola: cada uma tem o seu perfil, os seus modelos e a sua submissão, em
# modelos/<instância>/ e evaluation/results/posse_ml_<instância>.csv.
#   viterbi  rastro do Viterbi do tracking (tracking_bola/out/viterbi_bola.csv, em pixels)
#   fusao    fusão ATD + Roboflow (data/tactical/<jogo>_tactical_ball_tracker.csv)
#   atd      a bola do próprio ATD
INSTANCIAS = ("viterbi", "fusao", "atd")
INSTANCIA_PADRAO = "viterbi"
MODELOS = os.path.join(MODELOS_BASE, INSTANCIA_PADRAO)


def nome_paradas(caminho):
    """Nome curto de uma fonte de bola morta, para nomes de arquivo: gt, nuno ou o nome do CSV."""
    absoluto = os.path.abspath(caminho)
    if absoluto == os.path.abspath(GT_PUBLICO):
        return "gt"
    if absoluto == os.path.abspath(PARADAS_EQUIPE):
        return "nuno"
    return os.path.splitext(os.path.basename(caminho))[0]


def pasta_modelos(instancia, classificador="hgb"):
    """modelos/<instância> (hgb) ou modelos/<instância>/<classificador>. O perfil da bola fica sempre
    em modelos/<instância>/perfil_bola.json: é da bola, não do classificador."""
    base = os.path.join(MODELOS_BASE, instancia)
    return base if classificador == "hgb" else os.path.join(base, classificador)


def nome_saida(instancia, classificador="hgb", paradas=None, modo=None):
    """evaluation/results/posse_ml_<instância>[_<classificador>][_<fonte>_<modo>].csv"""
    nome = "posse_ml_" + instancia
    if classificador != "hgb":
        nome += "_" + classificador
    if paradas:
        nome += "_{}_{}".format(nome_paradas(paradas), modo)
    return os.path.join(RESULTADOS, nome + ".csv")


def argumento_classificador(ap):
    ap.add_argument("--classificador", default="hgb",
                    choices=("hgb", "xgboost", "lightgbm", "catboost", "gru", "lstm", "tcn", "transformer"),
                    help="hgb (padrão), xgboost, lightgbm, catboost, gru ou tcn; cada um tem a sua pasta "
                         "modelos/<instância>/<classificador>")


def bola_da_instancia(instancia):
    """O que passar como `bola` para carregar_jogo e caminho_bola."""
    return {"viterbi": None, "fusao": "fusao", "atd": "atd"}[instancia]


def argumento_instancia(ap):
    ap.add_argument("--instancia", choices=INSTANCIAS, default=INSTANCIA_PADRAO,
                    help="qual bola: viterbi (padrão), fusao ou atd. Define a bola, a pasta "
                         "modelos/<instância> e o nome da submissão; --bola e --modelos mudam só um deles")
DFL = os.environ.get("POSSE_ML_DFL") or os.path.join(PASTA, "dfl")
# os 3 jogos abertos da Metrica (baixar_metrica.py): Sample_Game_1, _2 e _3
METRICA = os.environ.get("POSSE_ML_METRICA") or os.path.join(PASTA, "metrica_dados")
GT_PUBLICO = os.environ.get("POSSE_ML_GT") or os.path.join(
    RAIZ, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

FPS = 25

# lados de cada jogador
MANDANTE, VISITANTE = 0, 1

# rótulos por frame, na mesma ordem do avaliador oficial
H, A, D, X = 0, 1, 2, 3
LETRAS = "HADX"

# Primeiro frame com bola em jogo em cada tempo (DATA.md e FAQ.md). O 2º tempo é avaliado desde o
# frame 110808, mas a bola só rola no 113742: os frames entre os dois estão todos mortos.
KICKOFFS = {"BAR-ATH": {1: 11185, 2: 113742}}


# Bola usada quando nenhum --bola é passado: o rastro do Viterbi do tracking (tracking_bola/
# track_temporal.py --method viterbi), em pixels do vídeo tático. Foi com ele que o posse/ teve o
# melhor spellF1 (posse/README.md, seção 1.8). A fusão continua disponível com --bola fusao.
BOLA_VITERBI = os.path.join(RAIZ, "tracking_bola", "out", "viterbi_bola.csv")


def caminho_bola(bola=None, jogo=None, pasta_dados=None):
    """Caminho do CSV da bola. bola: None (o padrão, $POSSE_ML_BOLA ou o Viterbi), "fusao"
    (data/tactical/<jogo>_tactical_ball_tracker.csv, saída do export_ball.py) ou um caminho."""
    if bola and bola != "fusao":
        return bola
    if bola is None:
        return os.environ.get("POSSE_ML_BOLA") or BOLA_VITERBI
    from games import GAME, asset_paths
    jogo = jogo or GAME
    pasta = os.path.dirname(asset_paths(jogo, pasta_dados)["atd_raw"])
    return os.path.join(pasta, "{}_tactical_ball_tracker.csv".format(jogo))


def ler_npz(caminho):
    """Todos os arrays de um .npz, com o arquivo fechado ao sair. No Windows, um .npz aberto
    impede outro processo de gravar por cima dele."""
    import numpy as np
    with np.load(caminho) as z:
        return {k: z[k] for k in z.files}


def gravar_npz(caminho, **arrays):
    """Grava num temporário e troca de nome. Se outro processo já gravou o mesmo arquivo (dois
    processos calculando a mesma coisa ao mesmo tempo), fica o dele."""
    import time
    import numpy as np
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    tmp = caminho[:-4] + ".{}.tmp.npz".format(os.getpid())
    np.savez(tmp, **arrays)
    for tentativa in range(10):
        try:
            os.replace(tmp, caminho)
            return
        except PermissionError:
            if os.path.isfile(caminho):
                break
            time.sleep(0.5 * (tentativa + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
    if not os.path.isfile(caminho):
        raise PermissionError("não consegui gravar {} (outro programa está com ele aberto?)".format(caminho))
