"""Três perguntas por frame:

  morta     a bola está fora de jogo agora?
  time      se está em jogo, o mandante está com ela?
  reinicio  se está parada, é o mandante quem vai repô-la?

Classificadores disponíveis (--classificador no treinar, calibrar e na submissão):

  hgb       HistGradientBoosting do scikit-learn (o padrão)
  xgboost   XGBoost, árvores por histograma, crescimento folha a folha como o hgb
  lightgbm  LightGBM
  catboost  CatBoost, árvores simétricas
  gru, lstm, tcn, transformer  redes de sequência sobre os mesmos atributos (sequencia.py)

Nos de árvore, cada pergunta tem o seu modelo, treinado só nas linhas em que ela faz sentido (o
"time" não vê bola parada; o "reinicio" só vê bola parada com reinício conhecido); os três formam um
"trio". Nas redes, uma rede só responde às três perguntas, com a perda de cada uma só nos frames em
que ela faz sentido. Os hiperparâmetros das árvores são os mesmos em espírito (400 iterações, taxa
0,05, ~63 folhas, folhas com bastante amostra), para a comparação ser do algoritmo e não do ajuste.
Tudo é salvo em modelos/.../conjuntos/<nome>/ com um manifesto da ordem das colunas e do
classificador, conferido ao carregar. Vários conjuntos são combinados pela média dos log-odds.
"""

import gc
import json
import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from .config import A, D, H, X

EPS = 1e-4
NOMES = ("morta", "time", "reinicio")
CLASSIFICADORES = ("hgb", "xgboost", "lightgbm", "catboost", "gru", "lstm", "tcn", "transformer")
SEQUENCIAIS = ("gru", "lstm", "tcn", "transformer")

HIPER = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
             l2_regularization=1.0, early_stopping=False)
HIPER_POR_CLASSIFICADOR = {
    "hgb": HIPER,
    "xgboost": dict(n_estimators=400, learning_rate=0.05, tree_method="hist", grow_policy="lossguide",
                    max_leaves=63, max_depth=0, min_child_weight=10.0, reg_lambda=1.0, max_bin=255),
    "lightgbm": dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                     reg_lambda=1.0, max_bin=255),
    "catboost": dict(iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=1.0),
}
# o nome do parâmetro com o número de árvores, para o --rapido
ITERACOES = {"hgb": "max_iter", "xgboost": "n_estimators", "lightgbm": "n_estimators",
             "catboost": "iterations"}


def _novo(classificador, semente, hiper, dispositivo):
    h = dict(HIPER_POR_CLASSIFICADOR[classificador], **(hiper or {}))
    # --hiper semente=N troca só o sorteio do modelo; o --semente do treinar.py também muda as cópias
    semente = int(h.pop("semente", semente))
    if classificador == "hgb":
        return HistGradientBoostingClassifier(random_state=semente, **h)
    if classificador == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=semente, n_jobs=-1, device=dispositivo, **h)
    if classificador == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=semente, n_jobs=-1, verbose=-1, deterministic=True,
                              force_row_wise=True, **h)
    if classificador == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=semente, verbose=0, thread_count=-1, allow_writing_files=False,
                                  task_type="GPU" if dispositivo == "cuda" else "CPU", **h)
    raise ValueError("classificador desconhecido: {}".format(classificador))


def _para_cpu(clf):
    """Modelo treinado na GPU passa a prever na CPU: a submissão tem que sair igual em qualquer máquina."""
    if type(clf).__name__ == "XGBClassifier":
        clf.set_params(device="cpu")
    return clf


def proximo_dono(rotulos):
    """Para cada frame de bola parada: 1 se o mandante repõe a bola, 0 se o visitante, -1 se não dá
    para saber (o próximo frame que não é bola parada está fora do jogo). Nos outros frames, -1."""
    r = np.asarray(rotulos)
    n = len(r)
    i = np.arange(n)
    # índice do primeiro frame em [i, n) que não é bola parada; n se não houver
    marco = np.where(r != D, i, n)
    proximo = np.minimum.accumulate(marco[::-1])[::-1]
    alvo = np.append(r, X)[proximo]
    out = np.where(alvo == H, 1, np.where(alvo == A, 0, -1)).astype(np.int8)
    return np.where(r == D, out, -1).astype(np.int8)


def _sem_coluna_vazia(x):
    """Coluna toda NaN quebra o HistGradientBoosting (sklearn 1.9). Vira 0 só na cópia de treino:
    constante, ela não gera nenhum corte, e o valor dela na previsão não importa."""
    vazia = ~np.isfinite(x).any(axis=0)
    if vazia.any():
        x = x.copy()
        x[:, vazia] = 0.0
    return x


def treinar(x, rotulos, proximo, semente=0, hiper=None, classificador="hgb", dispositivo="cpu"):
    """O trio de um classificador de árvores, nas linhas x (já com os frames escolhidos)."""
    viva = (rotulos == H) | (rotulos == A)
    morta = rotulos == D
    com_dono = morta & (proximo >= 0)

    def ajustar(linhas, alvo):
        # x[linhas] é uma cópia; quando são todas as linhas, não precisa
        sub = x if linhas.all() else x[linhas]
        clf = _novo(classificador, semente, hiper, dispositivo).fit(_sem_coluna_vazia(sub), alvo.astype(np.int8))
        del sub
        gc.collect()
        return _para_cpu(clf)

    return {
        "morta": ajustar(viva | morta, morta[viva | morta]),
        "time": ajustar(viva, rotulos[viva] == H),
        "reinicio": ajustar(com_dono, proximo[com_dono] == 1),
    }


def prever(trio, x, tempos=None):
    """(p_morta, p_mandante_com_bola_viva, p_mandante_reinicia) por frame. As redes de sequência
    precisam dos tempos (cada tempo é uma sequência); as árvores olham cada linha sozinha."""
    if hasattr(trio, "prever_partida"):
        return tuple(np.clip(p, EPS, 1 - EPS) for p in trio.prever_partida(x, tempos))
    return tuple(np.clip(trio[k].predict_proba(x)[:, 1], EPS, 1 - EPS) for k in NOMES)


def logit(p):
    return np.log(p) - np.log1p(-p)


def media_logit(ps):
    z = np.mean([logit(p) for p in ps], axis=0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)


def _versoes(classificador):
    import sklearn
    out = {"sklearn": sklearn.__version__}
    nome = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
            "gru": "torch", "lstm": "torch", "tcn": "torch", "transformer": "torch"}.get(classificador)
    if nome:
        out[nome] = __import__(nome).__version__
    return out


def salvar(trio, pasta, atributos, extra=None, classificador="hgb"):
    os.makedirs(pasta, exist_ok=True)
    if hasattr(trio, "salvar"):
        trio.salvar(pasta)
    else:
        for k in NOMES:
            with open(os.path.join(pasta, k + ".pkl"), "wb") as arq:
                pickle.dump(trio[k], arq, protocol=4)
    manifesto = {"atributos": list(atributos), "classificador": classificador}
    manifesto.update(_versoes(classificador))
    manifesto.update(extra or {})
    with open(os.path.join(pasta, "manifesto.json"), "w", encoding="utf-8") as arq:
        json.dump(manifesto, arq, indent=1, ensure_ascii=False)


def carregar(pasta, atributos=None):
    """O trio salvo em `pasta` (arquivos gerados por salvar(), só deste repositório)."""
    with open(os.path.join(pasta, "manifesto.json"), encoding="utf-8") as arq:
        manifesto = json.load(arq)
    if atributos is not None and list(atributos) != manifesto["atributos"]:
        raise RuntimeError("os modelos em {} foram treinados com outros atributos ({} contra {} "
                           "agora). Treine de novo: python -m posse_ml.treinar".format(
                               pasta, len(manifesto["atributos"]), len(atributos)))
    if manifesto.get("classificador") in SEQUENCIAIS:
        from . import sequencia
        return sequencia.carregar(pasta)
    trio = {}
    for k in NOMES:
        with open(os.path.join(pasta, k + ".pkl"), "rb") as arq:
            trio[k] = pickle.load(arq)
    return trio


def conjuntos(pasta=None):
    """Pastas de modelos/conjuntos/* que têm um trio salvo."""
    base = os.path.join(pasta or config.MODELOS, "conjuntos")
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, d) for d in sorted(os.listdir(base))
            if os.path.isfile(os.path.join(base, d, "manifesto.json"))]


def membros(pasta):
    """As pastas dos classificadores de uma combinação (membros.json, gravado pelo combinar.py), ou None."""
    caminho = os.path.join(pasta or config.MODELOS, "membros.json")
    if not os.path.isfile(caminho):
        return None
    with open(caminho, encoding="utf-8") as arq:
        relativos = json.load(arq)["pastas"]
    return [os.path.normpath(os.path.join(pasta, r)) for r in relativos]


def probabilidades(x, atributos, pasta=None, tempos=None):
    """Média dos conjuntos salvos: (p_morta, p_mandante, p_reinicio). Numa combinação, a média (em
    log-odds) das médias de cada classificador."""
    if membros(pasta):
        saidas = [probabilidades(x, atributos, m, tempos) for m in membros(pasta)]
        return tuple(media_logit([s[k] for s in saidas]) for k in range(3))
    pastas = conjuntos(pasta)
    if not pastas:
        raise IOError("nenhum modelo em {}. Treine com: python -m posse_ml.treinar".format(
            os.path.join(pasta or config.MODELOS, "conjuntos")))
    saidas = [prever(carregar(c, atributos), x, tempos) for c in pastas]
    return tuple(media_logit([s[k] for s in saidas]) for k in range(3))
