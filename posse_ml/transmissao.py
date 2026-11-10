"""Sincroniza o vídeo de TV com o tático e devolve, para cada frame tático, o que a TV mostrava.

Não há tabela de correspondência entre os dois vídeos, mas os dois filmam o mesmo jogo ao mesmo
tempo. Enquanto a TV mostra o jogo ao vivo, o frame da TV e o frame tático estão sempre à mesma
distância, uma por tempo (o intervalo foi editado de forma diferente nos dois). Replay e close não
correspondem a nada no tático, e é exatamente isso que interessa: a TV corta para eles quando o
jogo para.

Passos:
  1. distância grossa: o erro quadrático médio entre as duas bolas, para todas as distâncias de uma
     vez, por correlação cruzada com FFT; os dois mínimos a mais de um minuto um do outro são os
     dois tempos
  2. ajuste fino de ±10 frames em cada um, pela fração de frames com as bolas a menos de 1 m
  3. em blocos de 1 min da TV: qual dos dois tempos vale (um único ponto de corte, o que deixa os
     jogadores dos dois trackings mais próximos), e um ajuste de ±3 frames no bloco
  4. frame da TV ao vivo: a mediana, entre os jogadores da TV, da distância ao jogador mais próximo
     do mesmo time no tático fica abaixo de 1,5 m

Planos: AO_VIVO, OUTRA_VISTA (a homografia da TV existe, mas a imagem não bate com o tático: replay
do campo), SEM_CAMPO (close, torcida, gráfico) e SEM_TRANSMISSAO.
"""

import warnings

import numpy as np
import pandas as pd

from .atributos import compactar
from .config import FPS, MANDANTE, VISITANTE
from .janelas import fracao, velocidade

SEM_TRANSMISSAO, AO_VIVO, OUTRA_VISTA, SEM_CAMPO = -1, 0, 1, 2

DIST_AO_VIVO = 1.5
MIN_JOGADORES = 4
SEPARACAO = 1500             # frames entre os deslocamentos de tempos diferentes
JANELA_LOCAL = 1500
RAIO_LOCAL = 3
MIN_SOBREPOSICAO = 0.2       # fração dos frames com bola em movimento que precisam se sobrepor
MIN_RAPIDEZ = 1.0            # m/s: só bola em movimento entra na busca grossa
BLOCO = 20000


def _correlacao(a, b, n_fft):
    return np.fft.irfft(np.fft.rfft(a, n_fft) * np.conj(np.fft.rfft(b, n_fft)), n_fft)


def erro_por_deslocamento(bola_t, bola_b):
    """(deslocamentos, erro quadrático médio, sobreposição), com tático = transmissão + d.

    Só entram frames com a bola andando: trechos de bola parada (o tempo antes de um kickoff, a
    bola na marca do pênalti) são parecidos em qualquer momento do jogo e casariam deslocamentos
    falsos. Pelo mesmo motivo, um deslocamento só vale se os dois vídeos se sobrepõem numa boa
    parte do jogo (MIN_SOBREPOSICAO): o certo cobre um tempo inteiro, o falso, poucos minutos.
    """
    def andando(bola):
        v = velocidade(bola.astype(np.float64), 2, FPS)
        with np.errstate(invalid="ignore"):
            return np.isfinite(bola[:, 0]) & (np.hypot(v[:, 0], v[:, 1]) > MIN_RAPIDEZ)
    mt = andando(bola_t).astype(np.float64)
    mb = andando(bola_b).astype(np.float64)
    bola_t = np.where(mt[:, None] > 0, bola_t, np.nan)
    bola_b = np.where(mb[:, None] > 0, bola_b, np.nan)
    t = np.nan_to_num(bola_t.astype(np.float64))
    b = np.nan_to_num(bola_b.astype(np.float64))
    n_fft = 1 << int(np.ceil(np.log2(len(t) + len(b))))
    # soma de m_b(i) m_t(i+d) |t(i+d) - b(i)|^2 = sum m_b |b|^2 m_t + m_t |t|^2 m_b - 2 t.b
    conta = _correlacao(mt, mb, n_fft)
    termo_t = _correlacao(mt * (t ** 2).sum(axis=1), mb, n_fft)
    termo_b = _correlacao(mt, mb * (b ** 2).sum(axis=1), n_fft)
    cruzado = _correlacao(t[:, 0], b[:, 0], n_fft) + _correlacao(t[:, 1], b[:, 1], n_fft)
    soma = termo_t + termo_b - 2.0 * cruzado
    d = np.arange(n_fft)
    d = np.where(d > n_fft // 2, d - n_fft, d)
    conta = np.round(conta)
    with np.errstate(invalid="ignore", divide="ignore"):
        minimo = MIN_SOBREPOSICAO * min(mt.sum(), mb.sum()) / 2   # metade: cada tempo é ~1/2 do jogo
        erro = np.where(conta >= max(minimo, 1000), soma / conta, np.inf)
    ordem = np.argsort(d)
    return d[ordem], erro[ordem], conta[ordem]


def _concordancia(bola_t, bola_b, quadros, d, raio=1.0):
    j = quadros + d
    ok = (j >= 0) & (j < len(bola_t))
    dist = np.hypot(*(bola_t[j[ok]] - bola_b[quadros[ok]]).T)
    dist = dist[np.isfinite(dist)]
    return (float(np.mean(dist < raio)), len(dist)) if len(dist) > 50 else (0.0, len(dist))


def achar_deslocamentos(bola_t, bola_b, n_tempos=2):
    d, erro, _ = erro_por_deslocamento(bola_t, bola_b)
    escolhidos = []
    erro = erro.copy()
    for _ in range(n_tempos):
        i = int(np.argmin(erro))
        if not np.isfinite(erro[i]):
            break
        escolhidos.append(int(d[i]))
        erro[np.abs(d - d[i]) <= SEPARACAO] = np.inf
    vistos = np.flatnonzero(np.isfinite(bola_b[:, 0]))
    finos = []
    for c in escolhidos:
        cands = np.arange(c - 10, c + 11)
        notas = [_concordancia(bola_t, bola_b, vistos, o)[0] for o in cands]
        finos.append(int(cands[int(np.argmax(notas))]))
    return sorted(finos)


def _desencontro(tat, tra, deslocamento, quadros=None):
    """Para cada frame `quadros` da TV (todos, se None): mediana, entre os jogadores da TV, da
    distância ao jogador mais próximo do mesmo time no frame tático correspondente, e quantos
    jogadores entraram na conta."""
    quadros = np.arange(tra.n) if quadros is None else np.asarray(quadros)
    custo = np.full(len(quadros), np.nan)
    conta = np.zeros(len(quadros), dtype=np.int64)
    zero = np.zeros((1, 1, 2), dtype=np.float32)
    for a in range(0, len(quadros), BLOCO):
        q = quadros[a:a + BLOCO]
        j = np.clip(q + deslocamento[a:a + BLOCO], 0, tat.n - 1)
        menores = []
        for time in (MANDANTE, VISITANTE):
            pb, _ = compactar(tra.jogadores[q], np.broadcast_to(zero, tra.jogadores[q].shape), tra.lado[q], time)
            pt, _ = compactar(tat.jogadores[j], np.broadcast_to(zero, tat.jogadores[j].shape), tat.lado[j], time)
            dif = pb[:, :, None, :] - pt[:, None, :, :]
            dist = np.hypot(dif[..., 0], dif[..., 1])
            dist = np.where(np.isfinite(dist), dist, np.inf).min(axis=2)
            menores.append(np.where(np.isfinite(dist), dist, np.nan))
        m = np.concatenate(menores, axis=1)
        conta[a:a + BLOCO] = np.isfinite(m).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            custo[a:a + BLOCO] = np.nanmedian(m, axis=1)
    return custo, conta


def _mediana(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _por_bloco(tat, tra, desl, passo=5):
    """Deslocamento de cada frame da TV, decidido em blocos de JANELA_LOCAL frames.

    Primeiro, qual tempo vale em cada bloco: com os blocos em ordem, um só ponto de corte entre um
    tempo e o seguinte, o que soma o menor desencontro dos jogadores (bloco sem dado conta 2,5 m).
    Depois, dentro do bloco, ±RAIO_LOCAL frames em volta, pela menor distância mediana entre as
    duas bolas, aceita só se ficar abaixo de 1 m com pelo menos 100 frames comparáveis.
    """
    blocos = [np.arange(a, min(tra.n, a + JANELA_LOCAL)) for a in range(0, tra.n, JANELA_LOCAL)]
    nota = np.full((len(desl), len(blocos)), 2.5)
    for h, o in enumerate(desl):
        amostra = np.concatenate([b[::passo] for b in blocos])
        c, k = _desencontro(tat, tra, np.full(len(amostra), o), amostra)
        c = np.where((k >= MIN_JOGADORES) & np.isfinite(c), np.minimum(c, 5.0), np.nan)
        ini = 0
        for w, b in enumerate(blocos):
            fim = ini + len(b[::passo])
            m = _mediana(c[ini:fim])
            if np.isfinite(m):
                nota[h, w] = m
            ini = fim
    qual = np.zeros(len(blocos), dtype=np.int64)
    comeco = 0
    for h in range(len(desl) - 1):
        # cortar antes do bloco w: blocos [comeco, w) no tempo h, [w, fim) no tempo h + 1
        opcoes = range(comeco + 1, len(blocos) + 1)
        total = [nota[h, comeco:w].sum() + nota[h + 1, w:].sum() for w in opcoes]
        corte = list(opcoes)[int(np.argmin(total))]
        qual[corte:] = h + 1
        comeco = corte
    vista_b = np.isfinite(tra.bola[:, 0])
    out = np.empty(tra.n, dtype=np.int64)
    for w, b in enumerate(blocos):
        base = desl[qual[w]]
        out[b] = base
        q = b[vista_b[b]]
        if len(q) < 100:
            continue
        melhor, valor = base, np.inf
        for c in range(base - RAIO_LOCAL, base + RAIO_LOCAL + 1):
            j = q + c
            ok = (j >= 0) & (j < tat.n)
            d = np.hypot(*(tat.bola[j[ok]] - tra.bola[q[ok]]).T)
            d = d[np.isfinite(d)]
            if len(d) >= 100 and np.median(d) < valor:
                melhor, valor = c, float(np.median(d))
        if valor < 1.0:
            out[b] = melhor
    return out


def planos(tat, tra, homografia_valida):
    """(plano por frame tático, deslocamentos por tempo). tat e tra são Partidas (a TV com a bola
    do ATD dela); homografia_valida (tra.n,) bool."""
    desl = achar_deslocamentos(tat.bola, tra.bola)
    if not desl:
        raise RuntimeError("não achei o deslocamento entre a transmissão e o tático")
    por_frame = _por_bloco(tat, tra, desl)
    custo, conta = _desencontro(tat, tra, por_frame)
    ao_vivo = (conta >= MIN_JOGADORES) & (custo < DIST_AO_VIVO)
    alvo = np.arange(tra.n) + por_frame
    ok = (alvo >= 0) & (alvo < tat.n)
    plano = np.full(tat.n, SEM_TRANSMISSAO, dtype=np.int8)
    tipo = np.where(ao_vivo, AO_VIVO, np.where(homografia_valida, OUTRA_VISTA, SEM_CAMPO))
    plano[alvo[ok]] = tipo[ok]
    return plano, desl


def do_jogo(tat, jogo=None, pasta_dados=None):
    """Plano por frame tático para o jogo do desafio, lendo a transmissão do disco."""
    from games import GAME, asset_paths
    from .partida import carregar_jogo
    jogo = jogo or GAME
    tra = carregar_jogo(jogo=jogo, pasta_dados=pasta_dados, fonte="broadcast", goleiros=False)
    hom = pd.read_csv(asset_paths(jogo, pasta_dados, "broadcast")["homography"], usecols=["frame", "valid"])
    valida = np.zeros(tra.n, dtype=bool)
    f = hom["frame"].to_numpy()
    ok = (f >= 0) & (f < tra.n)
    valida[f[ok]] = hom["valid"].to_numpy()[ok] == 1
    return planos(tat, tra, valida)


def evidencia(plano, raio=25):
    """As duas colunas que entram na bola morta: fração de ±1 s sem campo e com outra vista."""
    return {"sem_campo": fracao(plano == SEM_CAMPO, raio), "outra_vista": fracao(plano == OUTRA_VISTA, raio)}
