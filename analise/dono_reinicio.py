"""Quem cobra cada parada: o time da primeira posse depois do reinício.

    from dono_reinicio import estimar_donos, aplicar_donos
    donos = estimar_donos(paradas, players, time_slot, ball, rotulos=base, tags=tags)
    rotulos = aplicar_donos(rotulos_por_frame, donos)

Toda parada termina com um time cobrando, e a posse seguinte começa com esse time. É o elo entre o
detector de bola morta e o H/A: se a parada estiver certa mas o time errado, a posse inteira que vem
depois do reinício vira um erro. Um detector de H/A por proximidade (o baseline) erra justamente aí,
porque durante a parada o jogador mais perto da bola costuma ser o do time que a pôs para fora.

Sinais, na ordem em que decidem (o primeiro que tiver resposta vale):

    tag            o time de um set piece do Smart Tagging que cai na parada (só no jogo do desafio)
    cobrador       o jogador mais perto da bola em volta do reinício, a menos de COBRADOR_M; primeiro numa
                   janela estreita e, sem resposta, numa larga (o reinício estimado pode errar alguns s)
    ultimo_toque   bola fora: o OUTRO time de quem chutou a bola por último antes da saída (último salto
                   de velocidade da bola com um jogador a menos de TOQUE_M)
    posse_antes    falta/impedimento: o time com a posse antes do apito (quem sofre a falta costuma ter a
                   bola); gol: o OUTRO time (quem levou o gol dá a saída)

Nada aqui lê ground truth. A ordem e os parâmetros foram escolhidos nos jogos de amostra 1 e 2 da Metrica
(validacao_metrica/), que são outros jogos, como o RULES.md permite. Lá, com o reinício no frame certo,
o cobrador acerta 98-100% quando responde; com o reinício errado em até ±2 s, ~90%. O último toque
acerta 75-85% nas bolas fora; a posse antes do apito, ~70% nas faltas. Detalhes em
validacao_metrica/README.md.

paradas: DataFrame com `inicio` e `fim` (frame do reinício, exclusivo do D) e, opcional, `tipo`. Aceita
os tipos do detector do Nuno (paradas.detectar_paradas): lateral / linha_de_fundo -> saída; centro ->
gol; o resto decide-se pela posição da bola no início da parada.
"""

import numpy as np
import pandas as pd

FPS = 25
LENGTH, WIDTH = 105.0, 68.0

# Params
COBRADOR_M = 2.5            # o cobrador está a menos disto da bola ...
JANELAS_COBRADOR_S = ((-1.5, 0.3), (-3.0, 1.5))   # ... numa destas janelas em volta do reinício, em ordem
TOQUE_M = 2.0               # "tocou na bola": a menos de 2 m dela ...
SALTO_MS = 1.5              # ... num frame em que a velocidade da bola sobe mais que isto (m/s) -- um chute
JANELA_TOQUE_S = 6.0        # olha o último chute nos 6 s antes do início da parada
LINHA_M = 2.0               # bola a menos de 2 m de uma linha (ou fora) = saída pela linha
JANELA_LINHA_S = 3.0        # última bola visível nos 3 s antes do início da parada
JANELA_TAG_S = (-5.0, 2.0)  # set piece com start_frame em [inicio - 5 s, fim + 2 s]
SEGURAR_S = 2.0             # aplicar_donos: o dono vale nos primeiros 2 s depois do reinício

ORDEM = {"saida": ("tag", "cobrador", "ultimo_toque", "posse_antes"),
         "gol": ("tag", "cobrador", "posse_antes"),
         "apito": ("tag", "cobrador", "posse_antes")}

TIPO_NUNO = {"lateral": "saida", "linha_de_fundo": "saida", "centro": "gol"}


def outro(t):
    return {"H": "A", "A": "H"}.get(t)


def _proximos(players, time_slot, ball, a, b):
    """(frames, time, distância) do jogador mais perto da bola em [a, b), só onde há bola."""
    a, b = max(0, a), min(len(ball), b)
    if b <= a:
        return np.array([], int), np.array([], "<U1"), np.array([])
    bl = ball[a:b]
    ok = ~np.isnan(bl[:, 0])
    if not ok.any():
        return np.array([], int), np.array([], "<U1"), np.array([])
    d = np.linalg.norm(players[a:b][ok] - bl[ok][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    k = d.argmin(1)
    dist = d[np.arange(len(k)), k]
    vivo = np.isfinite(dist)
    return np.flatnonzero(ok)[vivo] + a, time_slot[k][vivo], dist[vivo]


def sinal_cobrador(players, time_slot, ball, fim, fps=FPS):
    """Time do jogador mais perto da bola em volta do reinício, se alguém estiver a < COBRADOR_M.
    Vota frame a frame entre os frames com alguém perto; retorna (time, fração do voto) ou (None, 0).
    Tenta as janelas de JANELAS_COBRADOR_S em ordem e para na primeira com resposta."""
    for lo, hi in JANELAS_COBRADOR_S:
        _, t, d = _proximos(players, time_slot, ball, fim + int(lo * fps), fim + int(hi * fps) + 1)
        t = t[d < COBRADOR_M]
        if len(t):
            h = np.mean(t == "H")
            return ("H", h) if h >= 0.5 else ("A", 1 - h)
    return None, 0.0


def sinal_ultimo_toque(players, time_slot, ball, inicio, fps=FPS):
    """Bola fora: o OUTRO time de quem deu o último chute antes do início da parada.

    Chute = frame em que a velocidade da bola sobe mais que SALTO_MS de um frame para o outro, com um
    jogador a < TOQUE_M. Só "o último jogador perto da bola" não serve: perto da linha há quase sempre
    um adversário colado, e esse sinal acertou ~50% nos jogos da Metrica, contra ~80% do chute."""
    a, b = max(1, inicio - int(JANELA_TOQUE_S * fps)), inicio + 1
    if b <= a:
        return None
    v = np.linalg.norm(np.diff(ball[a - 1:b], axis=0), axis=1) * fps     # m/s em cada frame de [a, b)
    salto = np.diff(v, prepend=np.nan)
    f, t, d = _proximos(players, time_slot, ball, a, b)
    k = f - a
    chute = (d < TOQUE_M) & (np.nan_to_num(salto[k]) > SALTO_MS)
    return outro(t[chute][-1]) if chute.any() else None


def bola_na_linha(ball, inicio, fps=FPS):
    """True se a última bola visível antes do início da parada está a < LINHA_M de uma linha ou fora."""
    a = max(0, inicio - int(JANELA_LINHA_S * fps))
    bl = ball[a:inicio + 1]
    ok = np.flatnonzero(~np.isnan(bl[:, 0]))
    if not len(ok):
        return False
    x, y = bl[ok[-1]]
    return bool(min(x, LENGTH - x, y, WIDTH - y) < LINHA_M)


def _tipo(row, ball, fps):
    t = str(row.get("tipo", "") or "")
    if t in ("saida", "gol", "apito"):
        return t
    if t in TIPO_NUNO:
        return TIPO_NUNO[t]
    return "saida" if bola_na_linha(ball, int(row["inicio"]), fps) else "apito"


def _tag(tags, inicio, fim, fps):
    """Time do set piece do Smart Tagging mais perto do reinício, dentro da parada. tags: lista de
    dicts {start_frame, team} com team em 'H'/'A' (ou None)."""
    if not tags:
        return None
    lo, hi = inicio + JANELA_TAG_S[0] * fps, fim + JANELA_TAG_S[1] * fps
    cand = [t for t in tags if lo <= t["start_frame"] <= hi and t.get("team") in ("H", "A")]
    if not cand:
        return None
    return min(cand, key=lambda t: abs(t["start_frame"] - fim))["team"]


def estimar_donos(paradas, players, time_slot, ball, rotulos=None, tags=None, fps=FPS, ordem=None):
    """Uma linha por parada: inicio, fim, tipo_dono, dono, fonte e o voto de cada sinal.

    players (n, k, 2) e ball (n, 2) em metros, NaN = sem posição; time_slot (k,) 'H'/'A'.
    rotulos: H/A por frame de quem já tem um palpite de posse (ex.: o baseline); serve ao sinal
    posse_antes. tags: set pieces do Smart Tagging com o time (carregar_tags)."""
    ordem = ordem or ORDEM
    time_slot = np.asarray(time_slot)
    rows = []
    for _, r in paradas.iterrows():
        ini, fim = int(r["inicio"]), int(r["fim"])
        tipo = _tipo(r, ball, fps)
        cob, cob_frac = sinal_cobrador(players, time_slot, ball, fim, fps)
        votos = {"tag": _tag(tags, ini, fim, fps),
                 "cobrador": cob,
                 "ultimo_toque": sinal_ultimo_toque(players, time_slot, ball, ini, fps)}
        antes = None
        if rotulos is not None:
            prev = rotulos[max(0, ini - 5 * fps):ini]
            prev = prev[np.isin(prev, ["H", "A"])]
            antes = str(prev[-1]) if len(prev) else None
        votos["posse_antes"] = outro(antes) if tipo == "gol" else antes
        dono, fonte = None, ""
        for s in ordem[tipo]:
            if votos.get(s) in ("H", "A"):
                dono, fonte = votos[s], s
                break
        rows.append(dict(inicio=ini, fim=fim, tipo_dono=tipo, dono=dono, fonte=fonte,
                         cobrador_frac=round(cob_frac, 2), **{"voto_" + k: v for k, v in votos.items()}))
    return pd.DataFrame(rows)


def aplicar_donos(rotulos, donos, segurar_s=SEGURAR_S, mascarar_parada=True, fps=FPS):
    """Cópia de rotulos (array de 'H'/'A'/'D'/'X') com:

      [inicio, fim)               'D' (mascarar_parada): durante a parada ninguém tem a bola, então quem
                                  está perto dela não pode votar no H/A de depois
      [fim, fim + segurar_s)      o dono do reinício

    Pensado para a evidência POR FRAME de um classificador de H/A, ANTES da suavização: a suavização
    centrada é que espalha o time errado da parada para dentro da posse seguinte. Funciona também
    sobre rótulos finais, como correção a posteriori."""
    out = np.array(rotulos, dtype="<U1").copy()
    n = len(out)
    for r in donos.itertuples():
        if mascarar_parada:
            out[max(0, r.inicio):min(n, r.fim)] = "D"
        if r.dono in ("H", "A"):
            out[r.fim:min(n, r.fim + int(segurar_s * fps))] = r.dono
    return out


# ---- jogo do desafio ---------------------------------------------------------------------------------

def carregar_tags():
    """Set pieces do Smart Tagging com o time em 'H'/'A' (viewer/data/smart_tagging.json + meta).

    CONFERIR: o campo Team de um set piece deveria ser o time que cobra. Vale olhar alguns no viewer
    (assistir ao vídeo para entender os dados é permitido) antes de confiar nesse sinal."""
    import json
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(os.path.dirname(here), "viewer", "data")
    with open(os.path.join(data, "smart_tagging.json"), encoding="utf-8") as h:
        events = json.load(h)
    with open(os.path.join(data, "meta_tactical.json"), encoding="utf-8") as h:
        nomes = json.load(h).get("team_names", {})
    lado = {str(v).strip().lower(): k for k, v in nomes.items()}
    lado.update({"h": "H", "a": "A", "home": "H", "away": "A"})
    out = []
    for e in events:
        if (e.get("code") or "").upper() != "SET PIECES":
            continue
        out.append(dict(start_frame=int(e["start_frame"]), tipo=e.get("type"),
                        team=lado.get(str(e.get("team") or "").strip().lower())))
    return out


def donos_do_jogo(tabela, rotulos=None, ball=None, usar_tags=True):
    """Para a tabela de paradas.detectar_paradas: carrega o ATD e o Smart Tagging e estima os donos.
    ball: outra fonte de bola (fontes_bola.carregar_bola); por padrão a do ATD."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    viewer = os.path.join(os.path.dirname(here), "viewer")
    if viewer not in sys.path:
        sys.path.insert(0, viewer)
    from features import load_atd
    players, atd_ball, slot_teams, _ = load_atd()
    ball = atd_ball if ball is None else ball
    tags = carregar_tags() if usar_tags else None
    return estimar_donos(tabela, players, slot_teams, ball, rotulos=rotulos, tags=tags)
