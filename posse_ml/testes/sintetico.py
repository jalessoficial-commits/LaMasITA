"""Um jogo de mentira, com rótulos, para testar o pacote sem os dados do desafio nem da DFL.

Não é futebol de verdade, mas tem o que o pipeline precisa ver: o time com a bola avança e abre, o
outro recua e fecha, a bola vai de pé em pé no time que a tem, há trocas de posse num desarme e
paradas em que a bola sai, os jogadores andam e a bola fica parada até o reinício.

Também escreve esse jogo nos formatos de arquivo reais (ATD, CSV da bola, homografia, XML da DFL),
para testar a leitura e rodar o pipeline de ponta a ponta numa pasta temporária.
"""

import os

import numpy as np

from posse_ml.config import A, D, FPS, H, MANDANTE, VISITANTE, X
from posse_ml.partida import Partida

COMP, LARG = 105.0, 68.0
BASE = np.array([[5, 34], [20, 10], [20, 26], [20, 42], [20, 58], [40, 14], [40, 30], [40, 46],
                 [40, 60], [58, 24], [58, 44]], dtype=float)  # mandante atacando para a direita


def _formacao(ataca_direita, com_bola, avanco, comp, larg):
    pos = BASE.copy() * [comp / COMP, larg / LARG]
    if com_bola:
        pos[1:, 0] += avanco
        pos[1:, 1] = larg / 2 + (pos[1:, 1] - larg / 2) * 1.15
    else:
        pos[1:, 0] -= avanco * 0.5
        pos[1:, 1] = larg / 2 + (pos[1:, 1] - larg / 2) * 0.8
    if not ataca_direita:
        pos[:, 0] = comp - pos[:, 0]
    return pos


def _tempo(rng, n, direita_m, dono, comp, larg, espera=0):
    """Um tempo de n frames. Os `espera` primeiros são bola morta antes do pontapé inicial."""
    jog = np.full((n, 22, 2), np.nan, dtype=np.float32)
    bola = np.full((n, 2), np.nan, dtype=np.float32)
    alt = np.zeros(n, dtype=np.float32)
    rot = np.full(n, X, dtype=np.int8)
    alvo = {MANDANTE: _formacao(direita_m, False, 0, comp, larg),
            VISITANTE: _formacao(not direita_m, False, 0, comp, larg)}
    atual = {t: alvo[t].copy() for t in alvo}
    b_pos = np.array([comp / 2, larg / 2])
    for f in range(min(espera, n)):
        jog[f, :11], jog[f, 11:] = atual[MANDANTE], atual[VISITANTE]
        bola[f] = b_pos
        rot[f] = D
    f = espera
    portador = 6
    while f < n:
        for _ in range(rng.integers(2, 9)):
            dur = int(rng.uniform(1.0, 4.0) * FPS)
            novo = int(rng.integers(1, 11))
            for t in (MANDANTE, VISITANTE):
                ataca = direita_m if t == MANDANTE else not direita_m
                alvo[t] = _formacao(ataca, t == dono, rng.uniform(5, 20), comp, larg) + rng.normal(0, 3, (11, 2))
            ini = b_pos.copy()
            alto = rng.random() < 0.3
            for i in range(dur):
                if f >= n:
                    break
                for t in alvo:
                    atual[t] += (alvo[t] - atual[t]) * 0.03 + rng.normal(0, 0.05, (11, 2))
                meio = dur // 2
                if i < meio:
                    b_pos = ini + (atual[dono][portador] - ini) * (i / max(meio, 1))
                else:
                    b_pos = b_pos + (atual[dono][novo] - b_pos) * 0.3
                    if alto:
                        alt[f] = 4.0 * np.sin(np.pi * (i - meio) / max(dur - meio, 1))
                jog[f, :11], jog[f, 11:] = atual[MANDANTE], atual[VISITANTE]
                bola[f] = b_pos
                rot[f] = H if dono == MANDANTE else A
                f += 1
            portador = novo
        if f >= n:
            break
        if rng.random() < 0.5:
            outro = VISITANTE if dono == MANDANTE else MANDANTE
            portador = int(np.argmin(np.hypot(*(atual[outro][1:] - b_pos).T)) + 1)
            dono = outro
        else:
            saida = np.array([b_pos[0], -1.0 if b_pos[1] < larg / 2 else larg + 1.0])
            dur = int(rng.uniform(5, 20) * FPS)
            prox = MANDANTE if rng.random() < 0.5 else VISITANTE
            cobrador = int(rng.integers(1, 11))
            for i in range(dur):
                if f >= n:
                    break
                for t in alvo:
                    atual[t] += rng.normal(0, 0.02, (11, 2))
                # o cobrador vai até a bola
                atual[prox][cobrador] += (np.clip(saida, 0, [comp, larg]) - atual[prox][cobrador]) * 0.05
                jog[f, :11], jog[f, 11:] = atual[MANDANTE], atual[VISITANTE]
                bola[f] = saida
                rot[f] = D
                f += 1
            dono, portador = prox, cobrador
            b_pos = np.clip(saida, [0, 0], [comp, larg])
    return jog, bola, alt, rot


def jogo(segundos_por_tempo=600, semente=0, intervalo=500, layout=None, comp=COMP, larg=LARG):
    """Partida sintética. layout = (n_total, [(tempo, primeiro, último)], {tempo: kickoff}) para
    imitar a numeração de frames do jogo do desafio; sem ele, dois tempos seguidos."""
    rng = np.random.default_rng(semente)
    if layout is None:
        m = segundos_por_tempo * FPS
        n = 2 * m + intervalo
        tempos = [(1, 0, m - 1), (2, m + intervalo, n - 1)]
        kickoffs = None
    else:
        n, tempos, kickoffs = layout
    jog = np.full((n, 22, 2), np.nan, dtype=np.float32)
    bola = np.full((n, 2), np.nan, dtype=np.float32)
    alt = np.zeros(n, dtype=np.float32)
    rot = np.full(n, X, dtype=np.int8)
    for tempo, a, b in tempos:
        espera = (kickoffs or {}).get(tempo, a) - a
        partes = _tempo(rng, b - a + 1, tempo == 1, MANDANTE if tempo == 1 else VISITANTE, comp, larg, espera)
        jog[a:b + 1], bola[a:b + 1], alt[a:b + 1], rot[a:b + 1] = partes
    lado = np.broadcast_to(np.r_[np.zeros(11), np.ones(11)].astype(np.int8), (n, 22)).copy()
    return Partida(jogadores=jog, lado=lado, bola=bola, campo=(comp, larg), tempos=tempos,
                   kickoffs=kickoffs, rotulos=rot, altura_bola=alt, nome="sintetico{}".format(semente))


# ------------------------------------------------------------------------------------------------
# arquivos nos formatos reais
# ------------------------------------------------------------------------------------------------

def _fmt(v):
    return "NaN" if not np.isfinite(v) else "{:.5f}".format(v)


def escrever_atd(p, xml, txt, sem_bola=False):
    """Metadados (só times e jogadores) e texto do ATD, em coordenadas 0-1, frame_id desde 1."""
    comp, larg = p.campo
    with open(xml, "w") as arq:
        arq.write('<main><Metadata><Teams><Team id="TA"/><Team id="TB"/></Teams><Players>')
        for s, lado in enumerate(p.lado[0]):
            arq.write('<Player id="P{}" teamId="{}"/>'.format(s, "TA" if lado == MANDANTE else "TB"))
        arq.write("</Players></Metadata></main>")
    jog = p.jogadores / np.array([comp, larg], dtype=np.float32)
    bola = p.bola / np.array([comp, larg], dtype=np.float32)
    with open(txt, "w") as arq:
        for f in range(p.n):
            grupos = ["NaN" if not np.isfinite(x) else "{:.5f},{:.5f},0.0".format(x, y) for x, y in jog[f]]
            b = "NaN,NaN,NaN" if sem_bola or not np.isfinite(bola[f, 0]) else "{:.5f},{:.5f},NaN".format(*bola[f])
            arq.write("{}:{}:{}\n".format(f + 1, ";".join(grupos), b))


def escrever_bola_csv(bola_m, campo, caminho):
    norm = bola_m / np.array(campo, dtype=np.float32)
    with open(caminho, "w") as arq:
        arq.write("frame,x,y,fonte\n")
        for f, (x, y) in enumerate(norm):
            if np.isfinite(x):
                arq.write("{},{:.6f},{:.6f},detector\n".format(f, x, y))
            else:
                arq.write("{},,,sem_bola\n".format(f))


def homografia(camera, campo, foco=1500.0, imagem=(1920, 1080)):
    """Matriz 3x3 campo normalizado (0-1) -> pixels, de uma câmera em `camera` (metros) mirando o
    centro do campo."""
    comp, larg = campo
    c = np.asarray(camera, dtype=np.float64)
    alvo = np.array([comp / 2, larg / 2, 0.0])
    frente = (alvo - c) / np.linalg.norm(alvo - c)
    direita = np.cross(frente, [0, 0, 1.0])
    direita /= np.linalg.norm(direita)
    baixo = np.cross(frente, direita)
    rot = np.vstack([direita, baixo, frente])
    k = np.array([[foco, 0, imagem[0] / 2], [0, foco, imagem[1] / 2], [0, 0, 1]])
    t = -rot @ c
    hm = k @ np.column_stack([rot[:, 0], rot[:, 1], t])
    return hm @ np.diag([comp, larg, 1.0])


def escrever_homografia(caminho, n, matriz=None, valida=None):
    with open(caminho, "w") as arq:
        arq.write("frame," + ",".join("h{}{}".format(i, j) for i in range(3) for j in range(3))
                  + ",reprojection_error,valid\n")
        linha = ",".join("{:.10g}".format(v) for v in matriz.ravel()) if matriz is not None else ",".join(["0"] * 9)
        nada = ",".join(["NaN"] * 9)
        for f in range(n):
            ok = True if valida is None else bool(valida[f])
            arq.write("{},{},{},{}\n".format(f, linha if ok else nada, "0.2" if ok else "NaN", 1 if ok else 0))


def escrever_dfl(p, pasta, id_jogo):
    """Par de XML da DFL (informações e posições) para a Partida p, que precisa de rótulos."""
    comp, larg = p.campo
    with open(os.path.join(pasta, "DFL_02_01_matchinformation_DFL-COM-X_DFL-MAT-{}.xml".format(id_jogo)), "w") as arq:
        arq.write('<PutDataRequest><MatchInformation><Environment PitchX="{}" PitchY="{}"/><Teams>'
                  '<Team TeamId="CASA" Role="home"/><Team TeamId="FORA" Role="guest"/></Teams>'
                  '</MatchInformation></PutDataRequest>'.format(comp, larg))
    partes = {1: ("firstHalf", 10000), 2: ("secondHalf", 100000)}
    with open(os.path.join(pasta, "DFL_04_03_positions_raw_observed_DFL-COM-X_DFL-MAT-{}.xml".format(id_jogo)), "w") as arq:
        arq.write("<PutDataRequest><Positions>")
        for tempo, a, b in p.tempos:
            nome, base = partes[tempo]
            for s in range(p.jogadores.shape[1]):
                time = "CASA" if p.lado[a, s] == MANDANTE else "FORA"
                arq.write('<FrameSet GameSection="{}" TeamId="{}" PersonId="J{}">'.format(nome, time, s))
                for f in range(a, b + 1):
                    x, y = p.jogadores[f, s]
                    if np.isfinite(x):
                        arq.write('<Frame N="{}" X="{:.2f}" Y="{:.2f}" S="0"/>'.format(base + f - a, x - comp / 2, larg / 2 - y))
                arq.write("</FrameSet>")
            arq.write('<FrameSet GameSection="{}" TeamId="BALL" PersonId="BOLA">'.format(nome))
            for f in range(a, b + 1):
                x, y = p.bola[f]
                r = p.rotulos[f]
                arq.write('<Frame N="{}" X="{:.2f}" Y="{:.2f}" Z="{:.2f}" S="0" BallPossession="{}" BallStatus="{}"/>'.format(
                    base + f - a, x - comp / 2, larg / 2 - y, p.altura_bola[f], 1 if r == H else 2, 0 if r == D else 1))
            arq.write("</FrameSet>")
        arq.write("</Positions></PutDataRequest>")
