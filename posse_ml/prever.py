"""Do jogo carregado aos rótulos por frame, com os modelos e os ajustes salvos em modelos/.

  modelos/conjuntos/*/       os trios de classificadores (treinar.py)
  modelos/decodificador.json custos do Viterbi escolhidos na validação da DFL (treinar.py)
  modelos/calibracao.json    pesos da transmissão e adiantamento das paradas, ajustados na janela
                             pública (calibrar.py)
"""

import json
import os

import numpy as np

from . import atributos, config, modelos, transmissao, viterbi
from .config import FPS
from .janelas import fracao

# Sem calibracao.json: um valor de partida (1 s no começo da parada, 0,5 s no fim). O calibrar.py
# mede o atraso com a nossa bola na janela pública e grava o valor certo.
ADIANTAR = {"inicio": 25, "fim": 12}


def _json(nome, pasta):
    caminho = os.path.join(pasta, nome)
    if not os.path.isfile(caminho):
        return None
    with open(caminho, encoding="utf-8") as arq:
        return json.load(arq)


def arquivo_calibracao(paradas=None):
    """calibracao.json, ou calibracao_<fonte>.json quando as paradas de fora entram como evidência."""
    return "calibracao.json" if not paradas else "calibracao_{}.json".format(config.nome_paradas(paradas))


def ajustes(pasta=None, calibracao="calibracao.json"):
    """(custos do decodificador, pesos da evidência externa ou None, adiantamento)."""
    pasta = pasta or config.MODELOS
    dec = _json("decodificador.json", pasta)
    cal = _json(calibracao, pasta)
    if cal is None and calibracao != "calibracao.json":
        raise IOError("não achei {} em {}: rode antes o calibrar com o mesmo --paradas".format(calibracao, pasta))
    cal = cal or {}
    custos = dict(viterbi.CUSTOS, **((dec or {}).get("custos") or {}))
    return custos, cal.get("transmissao"), dict(ADIANTAR, **(cal.get("adiantar") or {}))


def ler_paradas(caminho, n):
    """(n,) 1 = bola morta, 0 = bola viva, -1 = sem informação. Aceita frame,label (H/A/D/X, como
    o GT ou uma submissão) e frame,morta (como o analise/out/paradas_D.csv)."""
    from .config import A, D, H
    from .medir_bola import ler_rotulos
    rot = ler_rotulos(caminho, n)
    return np.where(rot == D, 1, np.where((rot == H) | (rot == A), 0, -1)).astype(np.int8)


def evidencias(plano=None, forca=None):
    """As colunas de evidência externa de bola morta: plano da TV e, se houver, paradas de fora."""
    ev = transmissao.evidencia(plano) if plano is not None else {}
    if forca is not None:
        ev["paradas"] = fracao(forca == 1, FPS)
    return ev


def extra_da_transmissao(p_morta, plano, pesos, forca=None):
    """Quanto somar ao log-odds de bola morta: a regressão da calibração menos o próprio log-odds."""
    if pesos is None:
        return None
    ev = evidencias(plano, forca)
    nomes = [k for k in pesos if k not in ("intercepto", "modelo")]
    faltam = [k for k in nomes if k not in ev]
    if faltam:
        if plano is None and forca is None:
            return None
        raise ValueError("a calibração usa {}, que não foi dado agora".format(", ".join(faltam)))
    z = modelos.logit(p_morta)
    extra = pesos["intercepto"] + (pesos["modelo"] - 1.0) * z
    for nome in nomes:
        extra = extra + pesos[nome] * np.nan_to_num(ev[nome])
    return extra


def probabilidades(p, pasta=None, filtrar_pulos=True):
    x, nomes = atributos.matriz(p, filtrar_pulos=filtrar_pulos)
    return modelos.probabilidades(x, nomes, pasta, p.tempos)


def rotular(p, probs, plano=None, pasta=None, adiantar=True, custos=None, paradas=None, modo=None):
    """Códigos H/A/D/X por frame.

    paradas: CSV de bola morta de fora (GT, detector do Nuno...). modo "rigido": a bola morta é a
    do CSV onde ele diz algo, e o modelo só decide de quem é a bola (e quem repõe); sem adiantar as
    paradas, que já vêm prontas. modo "suave": o CSV entra como mais uma evidência de bola morta, com
    o peso ajustado pelo calibrar --paradas.
    """
    forca = ler_paradas(paradas, p.n) if paradas else None
    calibracao = arquivo_calibracao(paradas) if modo == "suave" else "calibracao.json"
    c, pesos, adi = ajustes(pasta, calibracao)
    c = dict(c, **(custos or {}))
    extra = extra_da_transmissao(probs[0], plano, pesos, forca if modo == "suave" else None)
    pontos = viterbi.emissoes(*probs, extra_morta=extra)
    rigido = modo == "rigido" and forca is not None
    rot = viterbi.rotulos(viterbi.decodificar(pontos, p.tempos, c, p.kickoffs, forca if rigido else None))
    if adiantar and not rigido:
        rot = viterbi.adiantar_paradas(rot, p.tempos, adi["inicio"], adi["fim"], p.kickoffs)
    return rot


def gravar(caminho, rot):
    letras = np.array(list(config.LETRAS))[rot]
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    with open(caminho, "w", newline="\n") as arq:
        arq.write("frame,label\n")
        arq.writelines("{},{}\n".format(i, letra) for i, letra in enumerate(letras))
    return caminho
