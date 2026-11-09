"""Pós-processamento do detector de paradas: fim no chute de reinício e paradas confirmadas pela cobrança.

    import eventos as E, paradas as P, reinicio as RE
    cfg = P.Config(v1_uniao=True)
    inicio = E.eventos_inicio(S, cfg.usar_inicio, fora=True)
    dead, tabela = P.detectar_paradas(S, cfg, inicio=inicio)
    dead, tabela = RE.aplicar(S, tabela, inicio, cfg)          # os dois passos abaixo

Não muda nada em paradas.py: entra depois dele, sobre a tabela de paradas. Nada aqui lê ground truth.

Passo 2 -- fim da parada no chute de reinício (refinar)
    O detector fecha muitas paradas onde a velocidade média dos PRÓXIMOS 5 s passa do limiar (ramo v1), e isso
    cai ~3,5 s antes do reinício. Um reinício é a bola parada que sai rápida com um jogador junto dela. Para
    cada parada, o fim passa ao primeiro chute em [fim - 4 s, fim + 5 s]. Sem chute visível, nas paradas
    fechadas pelo v1: a primeira arrancada coletiva (eventos_aceleracao) até 6 s depois, ou fim + 2 s.

Passo 3 -- paradas confirmadas pela cobrança (recuperar)
    Um terço das paradas perdidas tinha evento de início (bola saindo, apito), mas nada confirmou: em laterais e
    faltas cobradas rápido os jogadores não desaceleram. Para cada evento de início fora de qualquer D: se vier
    uma COBRANÇA (bola parada por 1,5 s e depois chutada, com um jogador junto) num lugar compatível com o
    tipo (lateral junto da linha lateral, linha de fundo junto do gol...), dentro do tempo máximo do tipo e
    sem nenhum outro chute no meio (isso seria jogo corrido), [evento, cobrança) vira D.

Validação (validacao_metrica/testar_detector_paradas.py): 3 jogos da Bundesliga com BallStatus por frame e
os 2 jogos da Metrica, bola apagada em 41% dos frames, apito e Smart Tagging emulados. Com o preditor do
Samuel, spellF1 +0,030 na Bundesliga e +0,015 na Metrica; positivo nos 5 jogos e em 47 de 50 rodadas com 5
preditores. A janela e o deslocamento foram escolhidos nesses jogos pelo viés do fim das paradas (~0 nos dois
conjuntos), não pelo spellF1. Detalhes em validacao_metrica/README.md.
"""

import numpy as np
import pandas as pd

FPS = 25

# Params -- passo 2
LENTA_MS = 2.5          # bola "parada": mediana da velocidade nos ANTES_S anteriores abaixo disto
RAPIDA_MS = 5.0         # bola "chutada": mediana nos DEPOIS_S seguintes acima disto
PERTO_M = 2.5           # algum jogador a menos disto da bola no frame do chute
ANTES_S, DEPOIS_S = 0.5, 0.3
JANELA_S = (-4.0, 5.0)  # onde procurar o chute, em volta do fim proposto pelo detector
ACEL_ATE_S = 6.0        # sem chute: arrancada coletiva até isto depois do fim do v1 ...
DESLOCA_V1_S = 2.0      # ... ou desloca o fim do v1 por isto
FUNDIR_S = 3.0          # como paradas.Config.fundir_s: buracos de "viva" menores que isto viram D

# Params -- passo 3
COBRANCA_ANTES_S = 1.5  # cobrança: bola parada (mediana < COBRANCA_LENTA_MS) por 1,5 s antes do chute
COBRANCA_LENTA_MS = 1.0
MAX_ATE_COBRANCA_S = {"lateral": 30, "linha_de_fundo": 45, "centro": 90, "apito": 45, "campo": 45,
                      "desconhecida": 45}
LATERAL_M, FUNDO_M, CENTRO_M = 3.0, 6.0, 3.0   # lugar compatível com o tipo da saída


def _vel_bola(ball):
    v = np.full(len(ball), np.nan)
    d = ball[2:] - ball[:-2]
    v[1:-1] = np.hypot(d[:, 0], d[:, 1]) * FPS / 2
    return v


def _perto(ball, players):
    with np.errstate(invalid="ignore"):
        d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
        return np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1) < PERTO_M


def _chutes(ball, players, antes_s, lenta_ms, min_antes, perto=None):
    v = pd.Series(_vel_bola(ball))
    a, b = int(antes_s * FPS), int(DEPOIS_S * FPS)
    antes = v.rolling(a, min_periods=min_antes).median().shift(1).values                      # [t - a, t)
    depois = v[::-1].rolling(b, min_periods=3).median()[::-1].shift(-1).values                 # (t, t + b]
    perto = _perto(ball, players) if perto is None else perto
    with np.errstate(invalid="ignore"):
        k = (antes < lenta_ms) & (depois > RAPIDA_MS) & perto
    return k & ~np.r_[False, k[:-1]]          # um frame por arrancada


def chutes(ball, players, perto=None):
    """Frames em que a bola, lenta nos 0,5 s antes, sai rápida com um jogador junto (bola e jogadores em m)."""
    return _chutes(ball, players, ANTES_S, LENTA_MS, 3, perto)


def cobrancas(ball, players, perto=None):
    """Chutes de uma bola PARADA por 1,5 s: a assinatura de uma bola recolocada em jogo."""
    return _chutes(ball, players, COBRANCA_ANTES_S, COBRANCA_LENTA_MS,
                   int(0.6 * COBRANCA_ANTES_S * FPS), perto)


def _mascara(intervalos, graded, base=None, fundir_s=FUNDIR_S):
    n = len(graded)
    dead = np.zeros(n, bool) if base is None else base.copy()
    for a, b in intervalos:
        dead[max(0, int(a)):min(n, int(b))] = True
    pad = np.concatenate(([False], ~dead, [False]))
    st = np.flatnonzero(~pad[:-1] & pad[1:])
    en = np.flatnonzero(pad[:-1] & ~pad[1:])
    for s, e in zip(st, en):
        if 0 < s and e < n and e - s < fundir_s * FPS:
            dead[s:e] = True
    return dead & graded


def refinar(tabela, ball, players, graded, aceleracoes=(), janela=JANELA_S, desloca_v1_s=DESLOCA_V1_S,
            k=None):
    """Passo 2. tabela: saída de paradas.detectar_paradas (inicio, fim, fonte_fim). Devolve (dead, tabela)."""
    k = np.flatnonzero(chutes(ball, players)) if k is None else k
    ac = np.asarray(sorted(aceleracoes), dtype=int)
    t = tabela.copy()
    if not len(t):
        return np.zeros(len(graded), bool), t
    fins, fontes = [], []
    for r in t.itertuples():
        fim, fonte = int(r.fim), r.fonte_fim
        lo = max(int(r.inicio) + 2 * FPS, fim + int(janela[0] * FPS))
        c = k[(k >= lo) & (k <= fim + int(janela[1] * FPS))]
        if len(c):
            fim, fonte = int(c[0]), "chute"
        elif r.fonte_fim == "v1":
            a = ac[(ac >= fim) & (ac <= fim + int(ACEL_ATE_S * FPS))] if len(ac) else []
            if len(a):
                fim, fonte = int(a[0]), "aceleracao_pos_v1"
            elif desloca_v1_s:
                fim, fonte = fim + int(desloca_v1_s * FPS), "v1_deslocado"
        fins.append(fim)
        fontes.append(fonte)
    t["fim"], t["fonte_fim"] = fins, fontes
    t["dur_s"] = (t.fim - t.inicio) / FPS
    return _mascara(zip(t.inicio, t.fim), graded), t


def lugar_compativel(tipo, x, y, length, width):
    if np.isnan(x):
        return False
    if tipo == "lateral":
        return min(y, width - y) < LATERAL_M
    if tipo == "linha_de_fundo":
        return min(x, length - x) < FUNDO_M
    if tipo == "centro":
        return np.hypot(x - length / 2, y - width / 2) < CENTRO_M
    return True


def recuperar(dead, inicio_eventos, ball, players, graded, length, width, guarda=True, k=None, ks=None):
    """Passo 3. inicio_eventos: eventos.eventos_inicio (já filtrados pelo SNR do apito). Devolve
    (dead, lista de paradas novas (inicio, fim, tipo))."""
    perto = _perto(ball, players)
    ks = np.flatnonzero(cobrancas(ball, players, perto)) if ks is None else ks
    todos = np.flatnonzero(chutes(ball, players, perto)) if k is None else k
    d = dead.copy()
    novas = []
    for ev in inicio_eventos:
        f0 = int(ev["frame"])
        if d[f0] or not graded[f0]:
            continue
        tipo = "apito" if ev["fonte"] == "apito" else ev.get("tipo", "desconhecida")
        hi = f0 + int(MAX_ATE_COBRANCA_S.get(tipo, 45) * FPS)
        c = [c for c in ks[(ks > f0 + int(1.5 * FPS)) & (ks <= hi)]
             if lugar_compativel(tipo, ball[c, 0], ball[c, 1], length, width)]
        if not c:
            continue
        fim = int(c[0])
        if guarda and np.any((todos > f0 + 2 * FPS) & (todos < fim - FPS)):
            continue                          # passes no meio: era jogo corrido
        if d[f0:fim].any():
            continue                          # o detector já viu esta parada de outro jeito
        d[f0:fim] = True
        novas.append((f0, fim, tipo))
    return _mascara([], graded, base=d), novas


def aplicar(S, tabela, inicio_eventos, cfg, length=None, width=None):
    """Os dois passos com os sinais do detector (eventos.Sinais): S.ball, S.players, S.graded.

    inicio_eventos: os MESMOS eventos passados a detectar_paradas (o filtro de SNR do apito é refeito aqui)."""
    import eventos as E
    from features import LENGTH, WIDTH
    length, width = length or LENGTH, width or WIDTH
    ball = np.asarray(S.ball, dtype=float)
    players = np.asarray(S.players, dtype=float)
    perto = _perto(ball, players)
    k = np.flatnonzero(chutes(ball, players, perto))
    ac = [a["frame"] for a in E.eventos_aceleracao(S)]
    dead, t = refinar(tabela, ball, players, S.graded, aceleracoes=ac, k=k)
    inicio = [e for e in inicio_eventos if e["fonte"] != "apito" or e["forca"] >= cfg.apito_min_snr]
    dead, novas = recuperar(dead, inicio, ball, players, S.graded, length, width, k=k)
    if novas:
        extra = pd.DataFrame([dict(inicio=a, fim=b, dur_s=(b - a) / FPS, tipo=tp, fonte_inicio="evento",
                                   fonte_fim="cobranca", confirmacao="cobranca") for a, b, tp in novas])
        t = pd.concat([t, extra], ignore_index=True).sort_values("inicio").reset_index(drop=True)
    return dead, t
