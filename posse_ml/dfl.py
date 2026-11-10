"""Leitor dos sete jogos abertos da DFL (Sportec Open Data), já com o rótulo de cada frame.

Fonte: Bassek, Rein, Weber e Memmert (2025), Scientific Data 12:195, licença CC-BY 4.0.

O que os torna úteis aqui é a bola: cada frame dela traz BallStatus (0 parada, 1 em jogo) e
BallPossession (1 mandante, 2 visitante), que juntos dão H, A ou D no mesmo sentido do gabarito do
desafio. São horas de jogo rotulado, contra 5 minutos de GT público.

Conversões: a DFL mede a partir do centro do campo com y para cima; aqui a origem vai para o canto
superior esquerdo, como no ATD. Os dois tempos são postos um depois do outro com INTERVALO frames
vazios entre eles, para que o resto do pacote trate os dois tipos de jogo do mesmo jeito. A leitura
usa só a biblioteca padrão (xml.etree): leva alguns minutos por jogo na primeira vez e depois sai do
cache.
"""

import glob
import hashlib
import os
import xml.etree.ElementTree as ET
from array import array

import numpy as np

from . import config
from .config import A, D, H, MANDANTE, VISITANTE, X
from .partida import Partida

INTERVALO = 500
PADRAO_POSICOES = "DFL_04_03_positions_raw_observed_*{}.xml"
PADRAO_INFO = "DFL_02_01_matchinformation_*{}.xml"


def jogos(pasta=None):
    """IDs dos jogos (DFL-MAT-...) com arquivo de posições na pasta."""
    pasta = pasta or config.DFL
    out = []
    for c in glob.glob(os.path.join(pasta, PADRAO_POSICOES.format(""))):
        out.append(os.path.basename(c).rsplit("DFL-MAT-", 1)[1][:-len(".xml")])
    return sorted(out)


def _arquivo(padrao, id_jogo, pasta):
    achados = glob.glob(os.path.join(pasta, padrao.format(id_jogo)))
    if not achados:
        raise IOError("não achei {} em {}".format(padrao.format(id_jogo), pasta))
    return achados[0]


def _info(caminho):
    """({TeamId: lado}, (comprimento, largura))."""
    raiz = ET.parse(caminho).getroot()
    lados = {}
    for t in raiz.iter("Team"):
        lados[t.get("TeamId")] = MANDANTE if t.get("Role") == "home" else VISITANTE
    amb = raiz.find(".//Environment")
    campo = (105.0, 68.0)
    if amb is not None:
        campo = (float(amb.get("PitchX", 105.0)), float(amb.get("PitchY", 68.0)))
    return lados, campo


def _ler_posicoes(caminho, lados):
    """{(parte, pessoa, time): array} direto do XML. Bola: N, X, Y, Z, posse, status; jogador: N, X, Y.

    Os valores vão para um array('d') plano, e não para uma lista de tuplas: são milhões de frames
    por jogo, e tuplas de float do Python ocupariam umas cinco vezes mais memória.
    """
    series = {}
    chave, linhas, largura = None, array("d"), 3
    for evento, el in ET.iterparse(caminho, events=("start", "end")):
        tag = el.tag
        if tag == "FrameSet":
            if evento == "start":
                time = el.get("TeamId")
                chave = (el.get("GameSection"), el.get("PersonId"), time) if (
                    time in lados or time.upper() == "BALL") else None
                linhas = array("d")
                largura = 6 if chave is not None and time.upper() == "BALL" else 3
            else:
                if chave is not None and len(linhas):
                    series[chave] = np.frombuffer(linhas, dtype=np.float64).reshape(-1, largura).copy()
                linhas = array("d")
                el.clear()
        elif tag == "Frame" and evento == "end":
            if chave is not None:
                g = el.get
                if largura == 6:
                    linhas.extend((float(g("N")), float(g("X")), float(g("Y")), float(g("Z", "nan")),
                                   float(g("BallPossession", "nan")), float(g("BallStatus", "nan"))))
                else:
                    linhas.extend((float(g("N")), float(g("X")), float(g("Y"))))
            el.clear()
    return series


def carregar(id_jogo, pasta=None, usar_cache=True):
    pasta = pasta or config.DFL
    pos = _arquivo(PADRAO_POSICOES, id_jogo, pasta)
    info = _arquivo(PADRAO_INFO, id_jogo, pasta)
    tag = hashlib.sha1("{}|{}".format(os.path.basename(pos), os.path.getsize(pos)).encode()).hexdigest()[:10]
    cache = os.path.join(config.CACHE, "dfl_{}_{}.npz".format(id_jogo, tag))
    if usar_cache and os.path.isfile(cache):
        z = config.ler_npz(cache)
        return Partida(jogadores=z["jogadores"], lado=z["lado"], bola=z["bola"],
                       campo=tuple(float(v) for v in z["campo"]),
                       tempos=[tuple(int(v) for v in t) for t in z["tempos"]],
                       rotulos=z["rotulos"], altura_bola=z["altura"], nome=id_jogo)

    lados, (comp, larg) = _info(info)
    series = _ler_posicoes(pos, lados)
    partes = sorted({k[0] for k in series}, key=lambda s: ("first" not in s.lower(), s))
    pessoas = sorted({(k[1], k[2]) for k in series if k[2].upper() != "BALL"},
                     key=lambda pt: (lados[pt[1]], pt[0]))
    coluna = {pt[0]: j for j, pt in enumerate(pessoas)}
    lado_slot = np.array([lados[t] for _, t in pessoas], dtype=np.int8)

    inicio, tempos, n = {}, [], 0
    for i, parte in enumerate(partes, start=1):
        quadros = np.concatenate([v[:, 0] for k, v in series.items() if k[0] == parte])
        lo, hi = int(quadros.min()), int(quadros.max())
        if i > 1:
            n += INTERVALO
        inicio[parte] = n - lo
        tempos.append((i, n, n + hi - lo))
        n += hi - lo + 1

    jog = np.full((n, len(pessoas), 2), np.nan, dtype=np.float32)
    bola = np.full((n, 2), np.nan, dtype=np.float32)
    altura = np.full(n, np.nan, dtype=np.float32)
    rot = np.full(n, X, dtype=np.int8)
    for (parte, pessoa, time), v in series.items():
        f = v[:, 0].astype(np.int64) + inicio[parte]
        x, y = v[:, 1] + comp / 2.0, larg / 2.0 - v[:, 2]
        if time.upper() == "BALL":
            bola[f, 0], bola[f, 1] = x, y
            altura[f] = v[:, 3]
            viva = v[:, 5] == 1
            rot[f] = np.where(viva, np.where(v[:, 4] == 1, H, A), D)
        else:
            jog[f, coluna[pessoa], 0], jog[f, coluna[pessoa], 1] = x, y
    lado = np.broadcast_to(lado_slot, (n, len(pessoas))).copy()

    if usar_cache:
        config.gravar_npz(cache, jogadores=jog, lado=lado, bola=bola, altura=altura, rotulos=rot,
                          campo=np.array([comp, larg]), tempos=np.array(tempos))
    return Partida(jogadores=jog, lado=lado, bola=bola, campo=(comp, larg), tempos=tempos,
                   rotulos=rot, altura_bola=altura, nome=id_jogo)


def inverter(p):
    """O mesmo jogo com mandante e visitante trocados, nos lados e nos rótulos."""
    rot = p.rotulos.copy()
    rot[p.rotulos == H] = A
    rot[p.rotulos == A] = H
    return p.trocar(lado=(1 - p.lado).astype(np.int8), rotulos=rot, nome=p.nome + "-invertido")
