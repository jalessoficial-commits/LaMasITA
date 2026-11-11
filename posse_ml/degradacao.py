"""Faz um jogo da DFL ficar com a cara do nosso: jogadores como no ATD, bola como no nosso tracker.

Treinar direto na DFL seria treinar com informação que não existe no jogo do desafio: bola em todo
frame, altura da bola, ninguém sumindo do tracking. O modelo aprenderia a contar com isso e erraria
aqui. Então cada cópia de treino recebe os defeitos do nosso jogo, em doses sorteadas por cópia, para
o modelo não se acostumar com uma dose só:

  jogadores     ruído que varia devagar na posição; slots que somem por alguns segundos
  bola alta     vai para o ponto do gramado na linha câmera-bola, que é onde a homografia a põe
  bola parada   na DFL ela fica congelada (às vezes fora do campo) e pula para o local do reinício no
                frame em que o jogo volta, o que entregaria a parada de bandeja ao modelo. Aqui ela
                desliza até parar, fica um tempo parada e é levada até onde o jogo recomeça.
  lacunas       a bola some com a frequência e a duração medidas no nosso tracker (perfil.py), de
                um jeito na bola em jogo (de preferência onde há muita gente em volta dela) e de
                outro na parada (quase toda ela, com as bordas soltas em volta do apito)
  pulos         a posição salta para outro ponto por alguns frames, na taxa, duração e distância
                medidas no nosso tracker
"""

import numpy as np
from scipy.signal import lfilter

from . import perfil as P
from .config import A, D, FPS, H
from .janelas import trechos

# Muda quando a degradação muda: entra na chave do cache das cópias de treino (treinar.py), para
# cópias feitas com uma versão antiga não serem reaproveitadas.
VERSAO = 2


def _ruido_lento(rng, forma, desvio, tau):
    """Ruído AR(1) com desvio marginal `desvio`, correlacionado por ~tau frames."""
    a = np.exp(-1.0 / tau)
    e = rng.normal(0.0, desvio * np.sqrt(1 - a * a), size=forma)
    e[0] = rng.normal(0.0, desvio, size=forma[1:])
    return lfilter([1.0], [1.0, -a], e, axis=0).astype(np.float32)


def sortear_parametros(rng, perfil, campo):
    f = perfil.get("falta") or P.PADRAO["falta"]
    par = {
        "ruido_jogador": rng.uniform(0.1, 0.3),
        "ruido_bola": rng.uniform(0.15, 0.4),
        "saidas_por_minuto": rng.uniform(0.2, 0.8),
        "falta_morta": float(np.clip((f["morta"] or 0.6) * rng.uniform(0.85, 1.15), 0.0, 0.98)),
        "falta_viva": float(np.clip((f["viva"] or 0.12) * rng.uniform(0.75, 1.25), 0.0, 0.9)),
        "taxa_pulo": perfil["pulos"]["taxa"] * rng.uniform(0.5, 1.5),
    }
    cam = perfil.get("camera")
    comp, larg = campo
    if cam:
        c = np.array(cam[str(rng.integers(1, len(cam) + 1))], dtype=np.float64)
        # a câmera medida está no campo do desafio; aqui ela vai para o mesmo lugar relativo
        par["camera"] = [c[0] * comp / 105.3 + rng.normal(0, 3.0), c[1] * larg / 68.0 + rng.normal(0, 3.0),
                         c[2] * rng.uniform(0.85, 1.15)]
    else:
        par["camera"] = [comp / 2.0, larg + rng.uniform(15.0, 45.0), rng.uniform(12.0, 30.0)]
    return par


def _bola_da_parada(bola, a, b, rng):
    """Uma bola plausível para a parada [a, b): rola freando, para, e é levada ao reinício."""
    m = b - a
    ini = bola[a - 1] if a > 0 else bola[a]
    fim = bola[b] if b < len(bola) else ini
    if not (np.all(np.isfinite(ini)) and np.all(np.isfinite(fim))):
        return np.full((m, 2), np.nan, dtype=np.float32)
    v0 = (bola[a - 1] - bola[a - 6]) * (FPS / 5.0) if a >= 6 else np.zeros(2)
    v0 = np.nan_to_num(v0)
    tau = rng.uniform(0.2, 0.8)                                  # s, freada
    t = np.arange(m) / FPS
    rolou = ini + v0 * tau * (1.0 - np.exp(-t / tau))[:, None]
    leva = min(m, int(rng.uniform(2.0, 6.0) * FPS))              # frames carregando até o reinício
    out = rolou.copy()
    if leva > 0:
        parada = rolou[m - leva]
        w = (np.arange(1, leva + 1) / float(leva + 1))[:, None]
        out[m - leva:] = parada + (fim - parada) * w
    return out.astype(np.float32)


def degradar(p, rng, perfil=None, par=None):
    """(Partida degradada, parâmetros sorteados). p precisa de rotulos (um jogo da DFL)."""
    perfil = perfil or P.PADRAO
    par = par or sortear_parametros(rng, perfil, p.campo)
    n, k = p.jogadores.shape[:2]
    rot = p.rotulos

    jog = p.jogadores + _ruido_lento(rng, p.jogadores.shape, par["ruido_jogador"], 12)
    minutos = n / (FPS * 60.0)
    for j in range(k):
        for _ in range(rng.poisson(par["saidas_por_minuto"] * minutos)):
            ini = int(rng.integers(0, n))
            # duração log-normal com mediana de 3 s, entre 1 s e 30 s
            dur = int(min(max(3 * FPS * np.exp(0.7 * rng.standard_normal()), FPS), 30 * FPS))
            jog[ini:ini + dur, j] = np.nan

    bola = p.bola.astype(np.float64).copy()
    morta = rot == D
    for a, b in trechos(morta):
        bola[a:b] = _bola_da_parada(bola, a, b, rng)

    # bola no ar: o ponto do chão no raio câmera -> bola
    z = np.nan_to_num(p.altura_bola if p.altura_bola is not None else np.zeros(n), nan=0.0)
    z = np.where(morta, 0.0, z)
    cx, cy, ch = par["camera"]
    sobe = np.clip(z / np.maximum(ch - z, 1.0), 0.0, 2.0)[:, None]
    bola = bola + sobe * (bola - np.array([cx, cy]))
    bola = bola + _ruido_lento(rng, bola.shape, par["ruido_bola"], 4)

    # As frações do perfil são medidas depois do filtro de pulos, que também tira posições. Parte da
    # falta vem dos pulos sorteados abaixo; as lacunas cobrem o resto.
    tp = min(par["taxa_pulo"], 0.5)
    alvo_morta = max(0.0, (par["falta_morta"] - tp) / (1.0 - tp))
    alvo_viva = max(0.0, (par["falta_viva"] - tp) / (1.0 - tp))

    oculto = np.zeros(n, dtype=bool)
    # Parada inteira escondida, com as bordas soltas em volta do apito: só quando o perfil não tem o
    # histograma das lacunas de bola morta (bola tipo ATD, que some na parada toda). Com o
    # histograma medido, as lacunas da parada saem dele (abaixo): a nossa bola some em pedaços
    # curtos na parada, não na parada inteira.
    paradas = trechos(morta) if not perfil["lacunas"].get("morta") else []
    if len(paradas):
        chance = min(1.0, alvo_morta / 0.9)
        for a, b in paradas:
            if rng.random() < chance:
                # some entre 0,5 s antes e 2,5 s depois do apito; volta entre 1,5 s antes e
                # 0,5 s depois do reinício
                atraso_ini, adianto_fim = (rng.uniform([-0.5, -0.5], [2.5, 1.5]) * FPS).astype(int)
                s, e = max(a + atraso_ini, 0), min(b - adianto_fim, n)
                if e > s:
                    oculto[s:e] = True
    viva = (rot == H) | (rot == A)
    # bola viva: lacunas onde a bola está cercada, com os tamanhos medidos
    perto = np.hypot(p.jogadores[..., 0] - p.bola[:, None, 0], p.jogadores[..., 1] - p.bola[:, None, 1])
    with np.errstate(invalid="ignore"):
        cerco = np.nansum(perto < 3.0, axis=1).astype(np.float64)
    # primeiro a bola morta, presa à parada com folga de 1 s, depois a viva, que já conta o que a
    # bola morta escondeu de bola viva em volta das paradas
    perto_da_parada = np.convolve(morta.astype(np.float64), np.ones(2 * FPS + 1), mode="same") > 0
    _completar(oculto, morta, alvo_morta, morta.astype(np.float64),
               P.lacunas_de(perfil, "morta"), rng, limite=perto_da_parada)
    _completar(oculto, viva, alvo_viva, np.where(viva, cerco ** 2 + 0.2, 0.0),
               P.lacunas_de(perfil, "viva"), rng)

    # pulos para outro objeto, só onde a bola continua visível
    visivel = ~oculto & np.isfinite(bola[:, 0])
    pul = perfil["pulos"]
    if par["taxa_pulo"] > 0 and visivel.any() and sum(pul["duracao"]) > 0:
        duracoes = P.sortear(rng, pul["bordas_duracao"], pul["duracao"], 4000).astype(int)
        quantos = int(par["taxa_pulo"] * visivel.sum() / max(duracoes.mean(), 1))
        inicios = rng.choice(np.flatnonzero(visivel), size=min(quantos, len(duracoes)), replace=False)
        dist = P.sortear(rng, pul["bordas_distancia"], pul["distancia"], len(inicios))
        ang = rng.uniform(0, 2 * np.pi, len(inicios))
        for ini, dur, r, t in zip(inicios, duracoes, dist, ang):
            fatia = slice(ini, min(n, ini + dur))
            bola[fatia] += np.array([r * np.cos(t), r * np.sin(t)])

    bola[oculto] = np.nan
    return p.trocar(jogadores=jog.astype(np.float32), bola=bola.astype(np.float32),
                    altura_bola=None, nome=p.nome + "-degradado"), par


def _completar(oculto, onde, alvo, peso, hist, rng, lote=500, limite=None):
    """Esconde lacunas sorteadas do histograma, centradas em frames sorteados por `peso`, até a
    fração escondida dos frames `onde` chegar a `alvo`. Com `limite`, só esconde dentro dele."""
    meta = alvo * onde.sum()
    if meta <= 0 or peso.sum() <= 0:
        return
    prob = peso / peso.sum()
    n = len(oculto)
    feito = int((oculto & onde).sum())
    for _ in range(200):
        centros = rng.choice(n, size=lote, p=prob)
        duracoes = np.maximum(P.sortear(rng, hist[0], hist[1], lote).astype(int), 1)
        for c, dur in zip(centros, duracoes):
            if feito >= meta:
                return
            s, e = max(0, c - dur // 2), min(n, c - dur // 2 + dur)
            novo = ~oculto[s:e] if limite is None else ~oculto[s:e] & limite[s:e]
            feito += int((novo & onde[s:e]).sum())
            oculto[s:e] |= novo
