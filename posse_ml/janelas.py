"""Estatísticas em janelas de frames, que ignoram NaN.

Todas usam somas acumuladas, então o tamanho da janela não muda o custo. "Antes" e "depois" não
incluem o próprio frame: são o que aconteceu nos w frames anteriores e nos w seguintes.
"""

import numpy as np


def trechos(mascara):
    """(m, 2) com [início, fim) de cada sequência de True."""
    m = np.asarray(mascara, dtype=bool)
    bordas = np.diff(np.concatenate(([False], m, [False])).astype(np.int8))
    return np.column_stack([np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)])


def _acumulado(x):
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    soma = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    conta = np.concatenate(([0], np.cumsum(ok)))
    return soma, conta


def _media(soma, conta, ini, fim):
    n = conta[fim] - conta[ini]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (soma[fim] - soma[ini]) / np.maximum(n, 1), np.nan)


def media_centrada(x, raio):
    """Média em [i - raio, i + raio]."""
    soma, conta = _acumulado(x)
    i = np.arange(len(soma) - 1)
    return _media(soma, conta, np.maximum(i - raio, 0), np.minimum(i + raio + 1, len(i)))


def media_antes(x, w):
    """Média em [i - w, i)."""
    soma, conta = _acumulado(x)
    i = np.arange(len(soma) - 1)
    return _media(soma, conta, np.maximum(i - w, 0), i)


def media_depois(x, w):
    """Média em (i, i + w]."""
    soma, conta = _acumulado(x)
    i = np.arange(len(soma) - 1)
    return _media(soma, conta, i + 1, np.minimum(i + w + 1, len(i)))


def fracao(mascara, raio):
    """Fração dos frames em [i - raio, i + raio] em que a máscara vale."""
    return media_centrada(np.asarray(mascara, dtype=np.float64), raio)


def desde(mascara, teto=None):
    """Frames desde a última vez que a máscara valeu (0 se vale agora). Sem nenhuma antes: teto."""
    m = np.asarray(mascara, dtype=bool)
    n = len(m)
    teto = n if teto is None else teto
    i = np.arange(n)
    ultimo = np.maximum.accumulate(np.where(m, i, -1))
    return np.where(ultimo >= 0, np.minimum(i - ultimo, teto), teto)


def ate(mascara, teto=None):
    """Frames até a próxima vez que a máscara vale (0 se vale agora). Sem nenhuma depois: teto."""
    return desde(np.asarray(mascara, dtype=bool)[::-1], teto)[::-1]


def deslocar(x, k):
    """y[i] = x[i + k], NaN onde i + k sai do array."""
    x = np.asarray(x, dtype=np.float64)
    y = np.full_like(x, np.nan)
    if k > 0:
        y[:-k] = x[k:]
    elif k < 0:
        y[-k:] = x[:k]
    else:
        y[:] = x
    return y


def velocidade(xy, raio, fps):
    """(x[i + raio] - x[i - raio]) * fps / (2 raio), em unidades por segundo. NaN nas pontas e onde
    faltar uma das duas posições. Serve para (n, 2) e para (n, k, 2)."""
    xy = np.asarray(xy)
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    if len(xy) > 2 * raio:
        v[raio:-raio] = (xy[2 * raio:] - xy[:-2 * raio]) * (fps / (2.0 * raio))
    return v
