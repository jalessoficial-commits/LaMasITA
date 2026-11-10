"""Uma partida em arrays, no mesmo formato para o jogo do desafio e para os jogos da DFL.

Coordenadas em metros a partir do canto superior esquerdo, x no comprimento do campo. O índice i
de todo array é o frame i do vídeo, que é também a linha i da submissão.

No jogo do desafio, os jogadores vêm do ATD e a bola, por padrão, do nosso tracker
(tracking_bola/export_ball.py), que substitui a bola do ATD em todos os frames.
"""

import dataclasses
import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config
from .config import FPS, MANDANTE, VISITANTE
from .janelas import velocidade

RAIO_VEL_JOGADOR = 5          # frames de cada lado na velocidade dos jogadores (0,2 s)
GOLEIRO_MIN_FRAMES = 1000     # frames presentes para um slot poder ser o goleiro
GOLEIRO_MIN_FRACAO = 0.5      # o goleiro é o mais recuado do time em pelo menos metade dos frames


@dataclasses.dataclass
class Partida:
    jogadores: np.ndarray            # (n, k, 2) float32; NaN = slot sem ninguém
    lado: np.ndarray                 # (n, k) int8, MANDANTE ou VISITANTE, frame a frame
    bola: np.ndarray                 # (n, 2) float32; NaN = sem bola
    campo: tuple                     # (comprimento, largura)
    tempos: list                     # [(tempo, primeiro_frame, ultimo_frame)] jogados, inclusive
    kickoffs: dict = None            # {tempo: frame em que a bola rola}; sem ele, o 1º do tempo
    rotulos: np.ndarray = None       # (n,) int8 H/A/D/X; só nos jogos da DFL
    altura_bola: np.ndarray = None   # (n,) metros; só nos jogos da DFL
    nome: str = ""

    @property
    def n(self):
        return len(self.bola)

    def kickoff(self, tempo):
        primeiro = dict((t, a) for t, a, _ in self.tempos)[tempo]
        return (self.kickoffs or {}).get(tempo, primeiro)

    def jogados(self):
        m = np.zeros(self.n, dtype=bool)
        for _, a, b in self.tempos:
            m[a:b + 1] = True
        return m

    def velocidades(self):
        """(n, k, 2) m/s por slot, diferença central de ±0,2 s."""
        return velocidade(self.jogadores, RAIO_VEL_JOGADOR, FPS)

    def trocar(self, **campos):
        return dataclasses.replace(self, **campos)


# ------------------------------------------------------------------------------------------------
# lados do campo
# ------------------------------------------------------------------------------------------------

def mandante_ataca_direita(p):
    """{tempo: True se o mandante ataca o gol de x = comprimento}, pela posição no kickoff.

    No pontapé inicial cada time está no próprio campo, então quem tem o x médio menor defende a
    esquerda. Usa o primeiro segundo depois do kickoff para não depender de um frame só.
    """
    out = {}
    for tempo, a, b in p.tempos:
        k = p.kickoff(tempo)
        k = k if a <= k <= b else a
        x = p.jogadores[k:k + FPS, :, 0]
        lado = p.lado[k:k + FPS]
        with np.errstate(invalid="ignore"):
            xm = np.nanmean(np.where(lado == MANDANTE, x, np.nan))
            xv = np.nanmean(np.where(lado == VISITANTE, x, np.nan))
        out[tempo] = bool(xm < xv)
    return out


def _goleiros(p, direita):
    """{lado: (slot, fração)}: em cada time, o slot que mais vezes é o jogador mais perto do próprio
    gol. Mesmo critério do diagnostico/verificar_goleiro.py."""
    k = p.jogadores.shape[1]
    vezes = np.zeros(k)
    presente = np.zeros(k)
    comp = p.campo[0]
    for tempo, a, b in p.tempos:
        x = p.jogadores[a:b + 1, :, 0].astype(np.float64)
        lado = p.lado[a:b + 1]
        presente += np.isfinite(x).sum(axis=0)
        for time in (MANDANTE, VISITANTE):
            defende_esquerda = direita[tempo] if time == MANDANTE else not direita[tempo]
            fundo = x if defende_esquerda else comp - x
            fundo = np.where((lado == time) & np.isfinite(fundo), fundo, np.inf)
            linhas = np.isfinite(fundo).any(axis=1)
            np.add.at(vezes, np.argmin(fundo[linhas], axis=1), 1)
    out = {}
    time_xml = p.lado[p.tempos[0][1]]
    for time in (MANDANTE, VISITANTE):
        frac = np.where((time_xml == time) & (presente >= GOLEIRO_MIN_FRAMES),
                        vezes / np.maximum(presente, 1), -1.0)
        slot = int(np.argmax(frac))
        out[time] = (slot, float(frac[slot]))
    return out


def corrigir_goleiros(p):
    """Lado dos dois slots de goleiro dado, frame a frame, pelo lado do campo em que eles estão.

    No ATD os slots dos goleiros trocam de pessoa: há trechos em que o slot do goleiro do Athletic
    segue o do Barcelona, com o time errado (diagnostico/README.md). Os outros slots ficam com o
    time do XML. Devolve (partida corrigida, {slot: frames corrigidos}).
    """
    direita = mandante_ataca_direita(p)
    goleiros = {t: s for t, (s, f) in _goleiros(p, direita).items() if f >= GOLEIRO_MIN_FRACAO}
    lado = p.lado.copy()
    mudou = {}
    meio = p.campo[0] / 2.0
    for tempo, a, b in p.tempos:
        for slot in goleiros.values():
            x = p.jogadores[a:b + 1, slot, 0]
            ok = np.isfinite(x)
            na_esquerda = x < meio
            # o mandante defende a esquerda quando ataca a direita
            dono = np.where(na_esquerda == direita[tempo], MANDANTE, VISITANTE).astype(np.int8)
            frames = a + np.flatnonzero(ok)
            mudou[slot] = mudou.get(slot, 0) + int(np.sum(lado[frames, slot] != dono[ok]))
            lado[frames, slot] = dono[ok]
    return p.trocar(lado=lado), mudou


# ------------------------------------------------------------------------------------------------
# leitura do jogo do desafio
# ------------------------------------------------------------------------------------------------

def times_dos_slots(caminho_xml):
    """MANDANTE/VISITANTE de cada slot. O primeiro time do XML é o mandante."""
    raiz = ET.parse(caminho_xml).getroot()
    ids = [t.get("id") for t in raiz.iter("Team")]
    if len(ids) != 2:
        raise ValueError("{}: esperava 2 times, achei {}".format(caminho_xml, len(ids)))
    return np.array([MANDANTE if j.get("teamId") == ids[0] else VISITANTE
                     for j in raiz.iter("Player")], dtype=np.int8)


def _par(texto):
    partes = texto.split(",")
    try:
        return float(partes[0]), float(partes[1])
    except (IndexError, ValueError):
        return np.nan, np.nan


def _ler_atd_texto(caminho, n_frames, n_slots):
    """Linha "<frame_id>:<slot>;...;<slot>:<bola>", cada grupo "x,y,s" ou NaN. frame_id começa em 1."""
    jog = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    bola = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(caminho) as arq:
        for linha in arq:
            partes = linha.rstrip("\n").split(":")
            if len(partes) != 3:
                continue
            f = int(partes[0]) - 1
            if not 0 <= f < n_frames:
                continue
            bola[f] = _par(partes[2])
            for s, grupo in enumerate(partes[1].split(";")[:n_slots]):
                if grupo and grupo[0] not in "Nn":
                    jog[f, s] = _par(grupo)
    return jog, bola


def _chave(caminho):
    st = os.stat(caminho)
    texto = "{}|{}|{}".format(os.path.abspath(caminho), st.st_size, int(st.st_mtime))
    return hashlib.sha1(texto.encode()).hexdigest()[:12]


def ler_atd(caminho, n_frames, n_slots, usar_cache=True):
    """(jogadores, bola) em coordenadas 0-1. O texto leva ~20 s; o resultado vai para o cache."""
    arq = os.path.join(config.CACHE, "atd_{}.npz".format(_chave(caminho)))
    if usar_cache and os.path.isfile(arq):
        z = config.ler_npz(arq)
        return z["jogadores"], z["bola"]
    jog, bola = _ler_atd_texto(caminho, n_frames, n_slots)
    if usar_cache:
        config.gravar_npz(arq, jogadores=jog, bola=bola)
    return jog, bola


def ler_bola_csv(caminho, n_frames, unidade=None):
    """Bola de um CSV com colunas frame (desde 0), x, y, em coordenadas 0-1 do campo; x vazio ou
    frame ausente = sem bola.

    unidade: "norm" (0-1, como o ATD e o export_ball.py), "px" (pixels do vídeo tático, como o
    tracking_bola/out/viterbi_bola.csv), ou None para descobrir pelos valores: acima de 2 só pode
    ser pixel. Pixel vira campo pela inversa da homografia, com o pixels_to_field do
    posse/predictor.py (a mesma conversão do export_ball.py).
    """
    tab = pd.read_csv(caminho, usecols=["frame", "x", "y"])
    bola = np.full((n_frames, 2), np.nan, dtype=np.float32)
    f = tab["frame"].to_numpy(dtype=np.int64)
    ok = (f >= 0) & (f < n_frames)
    bola[f[ok], 0] = tab["x"].to_numpy(dtype=np.float32)[ok]
    bola[f[ok], 1] = tab["y"].to_numpy(dtype=np.float32)[ok]
    if unidade is None:
        # a mediana, não o máximo: em 0-1 um pulo para fora do campo passa de 1, mas a bola típica
        # fica perto de 0,5; em pixels ela fica na casa das centenas
        unidade = "px" if np.nanmedian(np.abs(bola)) > 2.0 else "norm"
    if unidade == "px":
        import sys
        sys.path.insert(0, os.path.join(config.RAIZ, "posse"))
        import predictor
        bola = predictor.pixels_to_field(bola.astype(np.float64)).astype(np.float32)
        print("bola: {} em pixels, convertida pela homografia".format(os.path.basename(caminho)))
    elif unidade != "norm":
        raise ValueError("unidade da bola: norm ou px, não {}".format(unidade))
    return bola


def carregar_jogo(bola=None, jogo=None, pasta_dados=None, fonte="tactical", goleiros=True,
                  usar_cache=True):
    """O jogo do desafio como Partida.

    bola: None = a bola padrão (config.caminho_bola: o rastro do Viterbi do tracking), "fusao" = a
    fusão ATD + Roboflow do export_ball.py, "atd" = a bola do próprio ATD, ou o caminho de outro CSV
    frame,x,y (em 0-1 ou em pixels). Na fonte "broadcast" a bola é sempre a do ATD.
    """
    from games import GAME, GAMES, half_video_blocks, require, source_frames
    jogo = jogo or GAME
    xml, txt = require(["atd_meta", "atd_raw"], jogo, pasta_dados, fonte)
    time = times_dos_slots(xml)
    n = source_frames(fonte, jogo)
    jog, bola_atd = ler_atd(txt, n, len(time), usar_cache)
    if fonte != "tactical" or bola == "atd":
        b = bola_atd
    else:
        caminho = config.caminho_bola(bola, jogo, pasta_dados)
        if not os.path.isfile(caminho):
            raise IOError("não achei a bola em {}. Para o rastro do Viterbi: python "
                          "tracking_bola/track_temporal.py --method viterbi. Ou use --bola fusao, "
                          "--bola atd, ou --bola <outro CSV>.".format(caminho))
        b = ler_bola_csv(caminho, n)
    escala = np.array(GAMES[jogo]["pitch"], dtype=np.float32)
    tempos = half_video_blocks(jogo) if fonte == "tactical" else [(1, 0, n - 1)]
    p = Partida(jogadores=jog * escala, lado=np.broadcast_to(time, (n, len(time))).copy(),
                bola=b * escala, campo=tuple(float(v) for v in escala), tempos=tempos,
                kickoffs=config.KICKOFFS.get(jogo) if fonte == "tactical" else None,
                nome="{}-{}".format(jogo, fonte))
    if goleiros and fonte == "tactical":
        p, _ = corrigir_goleiros(p)
    return p
