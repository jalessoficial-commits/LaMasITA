"""Perfil do nosso tracker da bola: as estatísticas que a degradação da DFL tenta reproduzir.

O modelo é treinado na DFL, onde a bola é perfeita, e aplicado no nosso jogo, onde a bola vem do
tracker. Para que ele aprenda a lidar com a bola que vai receber, os jogos da DFL são estragados
até ficarem parecidos com ela (degradacao.py). "Parecidos" quer dizer, aqui:

  - quanto da bola falta com a bola viva e com a bola morta
  - quanto duram as lacunas, separadas em lacunas de bola viva e de bola morta
  - com que frequência a posição pula para outro objeto, por quantos frames e para quão longe
  - onde está a câmera tática (a bola no ar aparece no chão, no raio que sai da câmera)

São só agregados (frações e histogramas), sem nada frame a frame: o arquivo pode ir para o Git.
"""

import json

import numpy as np

from .bola import DIST_PULO, MIN_VIZINHOS, distancia_da_mediana
from .config import A, D, H
from .janelas import trechos

BORDAS_LACUNA = [1, 2, 3, 4, 6, 9, 13, 19, 25, 38, 50, 75, 100, 150, 225, 325, 500, 750, 1100,
                 1600, 2500, 5000]
BORDAS_DURACAO_PULO = [1, 2, 3, 4, 6, 9, 13]
BORDAS_DIST_PULO = [3, 4, 6, 9, 13, 19, 28, 40, 60, 90, 130]
MIN_LACUNAS = 30


def _hist(valores, bordas):
    return np.histogram(np.clip(valores, bordas[0], bordas[-1] - 1e-6), bins=bordas)[0].tolist()


def sortear(rng, bordas, contagens, tamanho=None):
    """Valores sorteados de um histograma: o bin pela contagem, o valor uniforme dentro dele."""
    c = np.asarray(contagens, dtype=np.float64)
    if c.sum() <= 0:
        raise ValueError("histograma vazio")
    b = np.asarray(bordas, dtype=np.float64)
    i = rng.choice(len(c), size=tamanho, p=c / c.sum())
    return rng.uniform(b[i], b[i + 1])


def medir(bola_m, tempos, rotulos=None, camera=None, descricao=None):
    """Perfil da bola (n, 2) em metros, nos frames dos tempos. rotulos (n,) H/A/D/X opcional, para
    separar bola viva de bola morta (o GT público ou uma predição do jogo todo)."""
    n = len(bola_m)
    avaliado = np.zeros(n, dtype=bool)
    for _, a, b in tempos:
        avaliado[a:b + 1] = True
    presente = np.isfinite(bola_m[:, 0]) & avaliado

    d, vizinhos = distancia_da_mediana(bola_m)
    with np.errstate(invalid="ignore"):
        pulo = presente & (vizinhos >= MIN_VIZINHOS) & (d > DIST_PULO)
    vista = presente & ~pulo

    lacunas = []
    for _, a, b in tempos:
        for s, e in trechos(~vista[a:b + 1]):
            lacunas.append((a + s, a + e))
    comp = np.array([e - s for s, e in lacunas])

    out = {
        "descricao": descricao or "",
        "frames_avaliados": int(avaliado.sum()),
        "presenca": float(presente.sum() / max(avaliado.sum(), 1)),
        "lacunas": {"bordas": BORDAS_LACUNA, "todas": _hist(comp, BORDAS_LACUNA)},
        "pulos": {
            "taxa": float(pulo.sum() / max(presente.sum(), 1)),
            "bordas_duracao": BORDAS_DURACAO_PULO,
            "duracao": _hist(np.diff(trechos(pulo), axis=1).ravel(), BORDAS_DURACAO_PULO),
            "bordas_distancia": BORDAS_DIST_PULO,
            "distancia": _hist(d[pulo], BORDAS_DIST_PULO),
        },
        "camera": camera,
    }

    if rotulos is not None:
        viva = avaliado & ((rotulos == H) | (rotulos == A))
        morta = avaliado & (rotulos == D)
        out["falta"] = {"viva": float((~vista[viva]).mean()) if viva.any() else None,
                        "morta": float((~vista[morta]).mean()) if morta.any() else None,
                        "frames_viva": int(viva.sum()), "frames_morta": int(morta.sum())}
        tipo = {"viva": [], "morta": []}
        for (s, e), L in zip(lacunas, comp):
            nv, nm = viva[s:e].sum(), morta[s:e].sum()
            if nv + nm == 0:
                continue
            tipo["viva" if nv >= nm else "morta"].append(L)
        for k, v in tipo.items():
            out["lacunas"]["n_" + k] = len(v)
            out["lacunas"][k] = _hist(v, BORDAS_LACUNA) if len(v) >= MIN_LACUNAS else None
    return out


# Sem perfil medido (só nos testes): números de ordem de grandeza do tracker, nada medido.
PADRAO = {
    "descricao": "padrão sem medição, só para testes",
    "presenca": 0.73,
    "falta": {"viva": 0.12, "morta": 0.6},
    "lacunas": {"bordas": BORDAS_LACUNA,
                "todas": [30, 25, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 2, 2, 1, 1]},
    "pulos": {"taxa": 0.05, "bordas_duracao": BORDAS_DURACAO_PULO, "duracao": [50, 20, 10, 6, 3, 1],
              "bordas_distancia": BORDAS_DIST_PULO, "distancia": [10, 12, 12, 10, 8, 6, 5, 3, 2, 1]},
    "camera": None,
}


def lacunas_de(perfil, tipo):
    """(bordas, contagens) das lacunas de bola viva ou morta, ou de todas se não houver medida."""
    lac = perfil["lacunas"]
    return lac["bordas"], lac.get(tipo) or lac["todas"]


def salvar(perfil, caminho):
    with open(caminho, "w", encoding="utf-8") as arq:
        json.dump(perfil, arq, indent=1, ensure_ascii=False)


def carregar(caminho):
    with open(caminho, encoding="utf-8") as arq:
        return json.load(arq)
