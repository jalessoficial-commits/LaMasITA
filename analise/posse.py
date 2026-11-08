"""Posse H/A a partir da bola e dos jogadores, com o D do detector de paradas por cima.

    import posse as PO
    lab = PO.prever(ball, players, slot_teams, dead, graded, PO.Config())

A ideia, em três passos:
1. controle por frame: o time do jogador mais próximo da bola (opcionalmente só quando ele está a
   < raio_m). Sem bola ninguém controla e a posse continua com quem tinha.
2. segmentação offline (Viterbi, o padrão): a sequência H/A que minimiza os frames em que o outro
   time controla + custo_troca por troca. Uma pressão de 1 s do adversário não paga uma troca; uma
   posse curta de verdade (3-4 s) paga. Dentro do D a troca é de graça, então o time depois da
   parada sai de quem joga depois dela (quem cobra), não de quem tinha antes.
   A alternativa online (histerese, cfg.custo_troca=None) foi pior: trocava no primeiro toque do
   reinício e partia as posses (analise/out/posse/).
3. D por cima (o detector v2).

Nos frames "detector" da bola fundida (sem ATD) a bola pode estar no ar, projetada longe de
quem a tem; com `so_atd_perto=True` esses frames só contam como controle se o jogador estiver bem
perto (raio_detector_m), já que uma bola no ar não costuma cair "em cima" de alguém por acaso.
"""

from dataclasses import dataclass

import numpy as np

FPS = 25


@dataclass
class Config:
    raio_m: float = float("inf")   # controle: jogador mais próximo a menos disso (inf = sempre)
    margem_m: float = 0.0          # e o mais próximo do outro time a mais de (d + margem)
    k: int = 5                     # frames de controle do outro time para trocar a posse
    janela: int = 25               # ... dentro de quantos frames
    raio_detector_m: float = 1.0   # raio nos frames só do detector (bola possivelmente no ar)
    so_atd_perto: bool = False
    reinicio_livre: bool = True    # depois de um D, o primeiro controle decide sem histerese
    inicio_padrao: str = "H"
    custo_troca: float = 50.0      # Viterbi: custo de uma troca de posse, em frames de evidência; None = histerese
    margem_peso_m: float = 0.0     # Viterbi: frame pesa min(1, |dH - dA| / margem); 0 = todos pesam 1


def distancias(ball, players, slot_teams):
    """(dH, dA): distância (m) do jogador mais próximo de cada time à bola, NaN sem bola."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    out = []
    for t in ("H", "A"):
        m = d[:, slot_teams == t]
        with np.errstate(all="ignore"):
            out.append(np.where(np.all(np.isnan(m), 1), np.nan, np.nanmin(np.where(np.isnan(m), np.inf, m), 1)))
    return out[0], out[1]


def controle(dH, dA, cfg, qualidade=None):
    """Código por frame: 0 = H controla, 1 = A controla, -1 = ninguém."""
    raio = np.full(len(dH), cfg.raio_m)
    if qualidade is not None and cfg.so_atd_perto:
        raio[qualidade == "detector"] = cfg.raio_detector_m
    h = (dH < raio) & ~(dA < dH + cfg.margem_m)
    a = (dA < raio) & ~(dH < dA + cfg.margem_m)
    c = np.full(len(dH), -1, np.int8)
    c[h] = 0
    c[a] = 1
    return c


def histerese(c, dead, cfg):
    """Máquina de estados: a posse troca quando o outro time soma >= k frames de controle nos últimos
    `janela` frames. Dentro do D o estado não muda; depois de um D o primeiro controle decide."""
    n = len(c)
    out = np.empty(n, np.int8)
    st = 0 if cfg.inicio_padrao == "H" else 1
    livre = False
    cnt = np.zeros(2, int)
    hist = np.full(max(cfg.janela, 1), -1, np.int8)
    for i in range(n):
        if dead[i]:
            livre = cfg.reinicio_livre
            cnt[:] = 0
            hist[:] = -1
            out[i] = st
            continue
        x = c[i]
        j = i % len(hist)
        old = hist[j]
        if old >= 0:
            cnt[old] -= 1
        hist[j] = x
        if x >= 0:
            cnt[x] += 1
            if livre:
                st, livre = x, False
                cnt[:] = 0
                hist[:] = -1
            elif x != st and cnt[x] >= max(cfg.k, 1):
                st = x
                cnt[:] = 0
                hist[:] = -1
        out[i] = st
    return out


def viterbi(c, dead, custo_troca, peso=None):
    """Segmentação offline em H/A que minimiza  (frames em que o outro time controla) + custo_troca
    x (número de trocas). Dentro do D não há evidência e a troca é de graça, então o time depois de
    uma parada sai dos dados que vêm depois dela (quem cobra), não de quem tinha antes.
    `peso` (opcional, por frame) pondera a evidência de cada frame de controle."""
    n = len(c)
    w = np.ones(n) if peso is None else peso
    em = np.zeros((n, 2))
    em[:, 0] = np.where(c == 1, w, 0.0)      # custo de dizer H quando A controla
    em[:, 1] = np.where(c == 0, w, 0.0)
    em[dead] = 0.0
    sw = np.where(dead, 0.0, custo_troca)    # custo de trocar ao entrar no frame i
    back = np.zeros((n, 2), bool)            # True = veio do outro estado
    v = em[0].copy()
    for i in range(1, n):
        s = sw[i]
        stay0, cross0 = v[0], v[1] + s
        stay1, cross1 = v[1], v[0] + s
        back[i, 0] = cross0 < stay0
        back[i, 1] = cross1 < stay1
        v = np.array([min(stay0, cross0) + em[i, 0], min(stay1, cross1) + em[i, 1]])
    out = np.empty(n, np.int8)
    st = int(np.argmin(v))
    for i in range(n - 1, -1, -1):
        out[i] = st
        if back[i, st]:
            st = 1 - st
    return out


def prever(ball, players, slot_teams, dead, graded, cfg=Config(), qualidade=None, dists=None):
    """Rótulos H/A/D por frame, 'X' fora dos tempos de jogo. cfg.custo_troca escolhe Viterbi
    (padrão) ou, com None, a histerese."""
    dH, dA = distancias(ball, players, slot_teams) if dists is None else dists
    c = controle(dH, dA, cfg, qualidade)
    if cfg.custo_troca is None:
        st = histerese(c, dead, cfg)
    else:
        peso = None
        if cfg.margem_peso_m > 0:
            peso = np.nan_to_num(np.clip(np.abs(dH - dA) / cfg.margem_peso_m, 0, 1))
        st = viterbi(c, dead, cfg.custo_troca, peso)
    lab = np.where(st == 0, "H", "A").astype("<U1")
    lab[dead] = "D"
    lab[~graded] = "X"
    return lab
