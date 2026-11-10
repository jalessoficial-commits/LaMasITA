"""Matriz de atributos: uma linha por frame, com as mesmas colunas no nosso jogo e na DFL.

O ATD não diz quem é quem (os slots trocam de pessoa e se partem no meio do jogo), então nenhuma
coluna depende de identidade, só do lado de cada jogador. Cada coluna responde a uma pergunta sobre
a cena: quem está mais perto da bola, há quanto tempo ela sumiu, como cada time está espalhado. Boa
parte delas é repetida olhando para trás e para a frente no tempo, porque a posse num frame se lê
melhor pelo que acontece em volta dele do que pelo frame sozinho.

Prefixo de cada grupo de colunas:

  perto_   dois jogadores mais próximos da bola em cada time; quantos a 2 m e a 5 m
  bola_    rapidez, aceleração e curva da bola; rapidez dela em relação ao mais próximo de cada time
  rumo_    em cada time, quem fica mais perto do ponto aonde a bola chega em 0,5 s e em 1 s
  lugar_   onde a bola está: no sentido em que o mandante ataca, longe do meio, além das linhas
  vis_     bola vista agora, há quantos frames, até quando, e em que fração de 1 s e de 4 s
  ritmo_   rapidez média dos jogadores (todos e por time), agora e em 2 s antes e depois
  toque_   proximidade do mandante menos a do visitante, em médias de 0,25, 1 e 3 s
  dom_     frames desde e até a última vez que cada time teve a bola dominada sem ninguém perto
  forma_   desenho de cada time no sentido do próprio ataque, marcação de um sobre o outro, e como
           o centro e o avanço de cada time mudam em 2 s
"""

import warnings

import numpy as np

from . import janelas as J
from .bola import limpar
from .config import FPS, MANDANTE, VISITANTE
from .partida import mandante_ataca_direita

TETO = 30 * FPS             # contagens de frames param em 30 s: além disso, a diferença não importa
ALCANCE_TOQUE = 1.5         # m, escala do núcleo de proximidade
DOMINIO_PERTO = 1.0         # m: o jogador está com a bola
DOMINIO_FOLGA = 2.5         # m: nenhum adversário a menos disso
DOMINIO_RAPIDEZ = 6.0       # m/s: acima disso é passe em voo, não domínio
FORA_MARGEM = 0.5           # m além da linha para a bola contar como fora
BLOCO = 40000               # frames por bloco nas contas (n, m, m)
SIGLAS = {MANDANTE: "m", VISITANTE: "v"}


def compactar(pos, vel, lado, time):
    """Jogadores de um time, em cada frame, nas primeiras colunas: (n, m, 2) posição e velocidade,
    NaN no resto. m é o maior número de jogadores desse time num frame."""
    ok = (lado == time) & np.isfinite(pos[..., 0])
    m = max(int(ok.sum(axis=1).max()), 1)
    ordem = np.argsort(~ok, axis=1, kind="stable")[:, :m]
    valido = np.take_along_axis(ok, ordem, axis=1)
    p = np.take_along_axis(pos, ordem[..., None], axis=1).astype(np.float32)
    v = np.take_along_axis(vel, ordem[..., None], axis=1).astype(np.float32)
    p[~valido] = np.nan
    v[~valido] = np.nan
    return p, v


def _distancias(pts, alvo):
    """(n, m) distância de cada jogador ao ponto alvo (n, 2); inf onde falta."""
    d = np.hypot(pts[..., 0] - alvo[:, None, 0], pts[..., 1] - alvo[:, None, 1])
    return np.where(np.isfinite(d), d, np.inf)


def _menor(d):
    out = d.min(axis=1)
    return np.where(np.isfinite(out), out, np.nan)


def _nucleo(d, alcance=ALCANCE_TOQUE):
    with np.errstate(invalid="ignore", over="ignore"):
        k = np.exp(-0.5 * (d / alcance) ** 2)
    return np.where(np.isfinite(k), k, 0.0)


def _nanstat(f, x, **kw):
    """f(x) sem os avisos de linha toda NaN (frame sem ninguém), que viram NaN mesmo."""
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return f(x, **kw)


def sentido_por_frame(p):
    """(n,) +1 onde o mandante ataca o gol de x = comprimento, -1 no outro, NaN fora dos tempos."""
    direita = mandante_ataca_direita(p)
    s = np.full(p.n, np.nan)
    for tempo, a, b in p.tempos:
        s[a:b + 1] = 1.0 if direita[tempo] else -1.0
    return s


def _forma(pos, vel, sinal, comp):
    """10 colunas da forma de um time, com x medido no sentido do ataque dele."""
    x = np.where(sinal[:, None] > 0, pos[..., 0], comp - pos[..., 0])
    y = pos[..., 1]
    avanco = vel[..., 0] * sinal[:, None]
    rapidez = np.hypot(vel[..., 0], vel[..., 1])
    n = np.isfinite(x).sum(axis=1)
    ordenado = np.sort(x, axis=1)                         # NaN vai para o fim
    # o mais recuado costuma ser o goleiro; a linha de defesa é o segundo
    ultima = np.where(n >= 2, ordenado[:, min(1, x.shape[1] - 1)], np.nan)
    frente = _nanstat(np.nanmax, x, axis=1)
    return {
        "media_x": _nanstat(np.nanmean, x, axis=1),
        "desvio_x": _nanstat(np.nanstd, x, axis=1),
        "desvio_y": _nanstat(np.nanstd, y, axis=1),
        "profundidade": frente - ultima,
        "ultima_linha": ultima,
        "frente": frente,
        "avanco": _nanstat(np.nanmean, avanco, axis=1),
        "rapidez": _nanstat(np.nanmean, rapidez, axis=1),
        "rapidez_p90": _nanstat(np.nanpercentile, rapidez, q=90, axis=1),
        "n": n.astype(np.float64),
    }


def _marcacao(de, para):
    """Média e mediana, sobre os jogadores de `de`, da distância ao jogador mais perto de `para`."""
    media = np.full(len(de), np.nan)
    mediana = np.full(len(de), np.nan)
    for a in range(0, len(de), BLOCO):
        b = min(len(de), a + BLOCO)
        dif = de[a:b, :, None, :] - para[a:b, None, :, :]
        d = _nanstat(np.nanmin, np.hypot(dif[..., 0], dif[..., 1]), axis=2)  # (bloco, m_de)
        media[a:b] = _nanstat(np.nanmean, d, axis=1)
        mediana[a:b] = _nanstat(np.nanmedian, d, axis=1)
    return media, mediana


def matriz(p, filtrar_pulos=True):
    """(X float32 (n, 102), nomes) para a Partida p."""
    bl = limpar(p.bola, filtrar=filtrar_pulos)
    bola, vel_bola = bl.pos.astype(np.float64), bl.vel.astype(np.float64)
    vel_jog = p.velocidades()
    comp, larg = p.campo
    sinal = sentido_por_frame(p)
    c = {}

    times = {t: compactar(p.jogadores, vel_jog, p.lado, t) for t in (MANDANTE, VISITANTE)}
    mais_perto = {}
    for t, (pos, vel) in times.items():
        s = SIGLAS[t]
        d = _distancias(pos, bola)
        dois = np.sort(d, axis=1)[:, :2] if d.shape[1] >= 2 else np.column_stack([d, np.full(len(d), np.inf)])
        c["perto_1" + s] = np.where(np.isfinite(dois[:, 0]), dois[:, 0], np.nan)
        c["perto_2" + s] = np.where(np.isfinite(dois[:, 1]), dois[:, 1], np.nan)
        c["perto_n2" + s] = (d < 2.0).sum(axis=1).astype(np.float64)
        c["perto_n5" + s] = (d < 5.0).sum(axis=1).astype(np.float64)
        idx = np.argmin(d, axis=1)
        vj = vel[np.arange(p.n), idx]
        tem = np.isfinite(d.min(axis=1))
        rel = np.hypot(*(vel_bola - vj).T)
        c["bola_rel" + s] = np.where(tem, rel, np.nan)
        mais_perto[t] = c["perto_1" + s]
    c["perto_saldo"] = c["perto_1m"] - c["perto_1v"]

    c["bola_rapidez"] = bl.rapidez.astype(np.float64)
    antes, depois = J.deslocar(vel_bola[:, 0], -3), J.deslocar(vel_bola[:, 0], 3)
    antes_y, depois_y = J.deslocar(vel_bola[:, 1], -3), J.deslocar(vel_bola[:, 1], 3)
    c["bola_acel"] = np.hypot(depois - antes, depois_y - antes_y) / (6.0 / FPS)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (antes * depois + antes_y * depois_y) / (np.hypot(antes, antes_y) * np.hypot(depois, depois_y))
        c["bola_curva"] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    for h in (0.5, 1.0):
        alvo = bola + vel_bola * h
        for t, (pos, _) in times.items():
            c["rumo_{:02d}{}".format(int(h * 10), SIGLAS[t])] = _menor(_distancias(pos, alvo))

    with np.errstate(invalid="ignore"):
        c["lugar_x"] = np.where(sinal > 0, bola[:, 0], comp - bola[:, 0])
        c["lugar_y"] = np.abs(bola[:, 1] - larg / 2.0)
        fora = np.fmax.reduce([-bola[:, 0], bola[:, 0] - comp, -bola[:, 1], bola[:, 1] - larg])
        c["lugar_fora"] = fora
        saiu = fora > FORA_MARGEM
    c["lugar_fora1s"] = J.fracao(saiu, FPS)
    c["lugar_fora_antes2s"] = J.media_antes(saiu.astype(np.float64), 2 * FPS)

    c["vis_agora"] = bl.vista.astype(np.float64)
    c["vis_desde"] = J.desde(bl.vista, TETO).astype(np.float64)
    c["vis_ate"] = J.ate(bl.vista, TETO).astype(np.float64)
    c["vis_1s"] = J.fracao(bl.vista, FPS)
    c["vis_4s"] = J.fracao(bl.vista, 4 * FPS)

    rapidez_jog = np.hypot(vel_jog[..., 0], vel_jog[..., 1])
    ritmos = {"": _nanstat(np.nanmean, rapidez_jog, axis=1)}
    for t, (_, vel) in times.items():
        ritmos[SIGLAS[t]] = _nanstat(np.nanmean, np.hypot(vel[..., 0], vel[..., 1]), axis=1)
    for s, r in ritmos.items():
        c["ritmo" + s] = r
        c["ritmo" + s + "_2s"] = J.media_centrada(r, 2 * FPS)
        c["ritmo" + s + "_antes"] = J.media_antes(r, 2 * FPS)
        c["ritmo" + s + "_depois"] = J.media_depois(r, 2 * FPS)

    saldo = np.where(bl.conhecida, _nucleo(mais_perto[MANDANTE]) - _nucleo(mais_perto[VISITANTE]), np.nan)
    for w in (6, 25, 75):
        c["toque_antes{}".format(w)] = J.media_antes(saldo, w)
        c["toque_depois{}".format(w)] = J.media_depois(saldo, w)

    with np.errstate(invalid="ignore"):
        devagar = np.where(np.isfinite(bl.rapidez), bl.rapidez < DOMINIO_RAPIDEZ, False) & bl.vista
        dm, dv = mais_perto[MANDANTE], mais_perto[VISITANTE]
        dominio = {MANDANTE: devagar & (dm < DOMINIO_PERTO) & (dv > DOMINIO_FOLGA),
                   VISITANTE: devagar & (dv < DOMINIO_PERTO) & (dm > DOMINIO_FOLGA)}
    for t, m in dominio.items():
        c["dom_desde" + SIGLAS[t]] = J.desde(m, TETO).astype(np.float64)
        c["dom_ate" + SIGLAS[t]] = J.ate(m, TETO).astype(np.float64)
    c["dom_desde_saldo"] = c["dom_desdem"] - c["dom_desdev"]
    c["dom_ate_saldo"] = c["dom_atem"] - c["dom_atev"]

    formas = {}
    por_time = {t: _forma(pos, vel, sinal if t == MANDANTE else -sinal, comp)
                for t, (pos, vel) in times.items()}
    for t, f in por_time.items():
        for k, v in f.items():
            formas["forma_{}{}".format(k, SIGLAS[t])] = v
    for k in por_time[MANDANTE]:
        formas["forma_{}_saldo".format(k)] = por_time[MANDANTE][k] - por_time[VISITANTE][k]
    pm, pv = times[MANDANTE][0], times[VISITANTE][0]
    mm = _marcacao(pm, pv)
    mv = _marcacao(pv, pm)
    formas["forma_marca_mediam"], formas["forma_marca_medianam"] = mm
    formas["forma_marca_mediav"], formas["forma_marca_medianav"] = mv
    formas["forma_marca_media_saldo"] = mm[0] - mv[0]
    formas["forma_marca_mediana_saldo"] = mm[1] - mv[1]
    for k in ("media_xm", "avancom", "media_xv", "avancov", "marca_media_saldo", "marca_mediana_saldo"):
        formas["forma_{}_antes".format(k)] = J.media_antes(formas["forma_" + k], 2 * FPS)
        formas["forma_{}_depois".format(k)] = J.media_depois(formas["forma_" + k], 2 * FPS)
    fora_dos_tempos = ~p.jogados()
    for k in formas:
        formas[k] = np.where(fora_dos_tempos, np.nan, formas[k])
    c.update(formas)

    nomes = list(c)
    x = np.empty((p.n, len(nomes)), dtype=np.float32)
    for j, k in enumerate(nomes):
        x[:, j] = c[k]
    return x, nomes
