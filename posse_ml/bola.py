"""Limpeza da bola, igual no jogo do desafio e nos jogos degradados da DFL.

A mesma função roda nas duas, para que o modelo veja no treino a bola que vai ver no jogo:

1. Filtro de pulos. Cerca de 1 em 10 posições do nosso tracker cai em outro objeto por poucos
   frames (tracking_bola/README.md). A posição a mais de 3 m da mediana, coordenada a coordenada,
   dos ±5 frames em volta é descartada (o mesmo filtro do posse/predictor.py).
2. Ponte de lacunas curtas: um buraco de até 8 frames (0,32 s) entre duas posições é preenchido em
   linha reta. Num buraco desse tamanho a bola não tem tempo de fazer muita curva; em buracos
   maiores a reta passa a inventar trajetória, então eles ficam vazios.
3. Velocidade: diferença entre 2 frames depois e 2 antes. Com um frame só, o tremido da posição
   pesa mais que o movimento.
"""

import dataclasses
import warnings

import numpy as np

from .config import FPS
from .janelas import trechos, velocidade

RAIO_PULO = 5
DIST_PULO = 3.0
MIN_VIZINHOS = 3
PONTE_MAX = 8
RAIO_VEL = 2


@dataclasses.dataclass
class BolaLimpa:
    pos: np.ndarray        # (n, 2) depois do filtro e da ponte
    vista: np.ndarray      # (n,) havia bola depois do filtro, antes da ponte
    conhecida: np.ndarray  # (n,) há posição depois da ponte
    vel: np.ndarray        # (n, 2) m/s
    rapidez: np.ndarray    # (n,) m/s
    pulo: np.ndarray       # (n,) posição descartada pelo filtro


def distancia_da_mediana(bola, raio=RAIO_PULO):
    """(distância de cada posição à mediana dos frames em [i - raio, i + raio], quantos desses
    frames têm bola). A janela inclui o próprio frame, como no posse/predictor.py."""
    bola = np.asarray(bola, dtype=np.float64)
    pad = np.pad(bola, ((raio, raio), (0, 0)), constant_values=np.nan)
    jan = np.lib.stride_tricks.sliding_window_view(pad, 2 * raio + 1, axis=0)  # (n, 2, w)
    vizinhos = np.isfinite(jan[:, 0, :]).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mediana = np.nanmedian(jan, axis=2)
    return np.hypot(*(bola - mediana).T), vizinhos


def remover_pulos(bola, raio=RAIO_PULO, dist=DIST_PULO, min_vizinhos=MIN_VIZINHOS):
    """(bola sem os pulos, máscara dos frames descartados)."""
    bola = np.asarray(bola, dtype=np.float64)
    d, vizinhos = distancia_da_mediana(bola, raio)
    with np.errstate(invalid="ignore"):
        fora = np.isfinite(bola[:, 0]) & (vizinhos >= min_vizinhos) & (d > dist)
    out = bola.copy()
    out[fora] = np.nan
    return out.astype(np.float32), fora


def ponte(bola, max_lacuna=PONTE_MAX):
    """Interpolação linear nas lacunas internas de até max_lacuna frames."""
    out = np.array(bola, dtype=np.float32)
    falta = ~np.isfinite(out[:, 0])
    for a, b in trechos(falta):
        if a == 0 or b == len(out) or b - a > max_lacuna:
            continue
        w = (np.arange(1, b - a + 1) / float(b - a + 1))[:, None]
        out[a:b] = out[a - 1] + (out[b] - out[a - 1]) * w
    return out


def limpar(bola, filtrar=True, max_lacuna=PONTE_MAX):
    if filtrar:
        sem_pulo, pulo = remover_pulos(bola)
    else:
        sem_pulo, pulo = np.asarray(bola, dtype=np.float32), np.zeros(len(bola), dtype=bool)
    vista = np.isfinite(sem_pulo[:, 0])
    pos = ponte(sem_pulo, max_lacuna)
    vel = velocidade(pos, RAIO_VEL, FPS)
    return BolaLimpa(pos=pos, vista=vista, conhecida=np.isfinite(pos[:, 0]), vel=vel,
                     rapidez=np.hypot(vel[:, 0], vel[:, 1]), pulo=pulo)
