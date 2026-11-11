"""Decodificador: transforma as probabilidades de cada frame numa sequência de posses.

Pegar o estado mais provável frame a frame não serve: a métrica que ranqueia (spellF1) compara
posses inteiras, e uma sequência que oscila entre H e A por alguns frames cria posses curtas que não
existem e quebra as verdadeiras. O Viterbi escolhe a sequência inteira de maior pontuação com uma
multa fixa por mudança de estado, e uma mudança só compensa quando a evidência acumulada depois dela
cobre a multa.

A bola parada vira dois estados, conforme quem vai repor, e cada um só leva à posse do próprio time.
Na prática, durante a parada o modelo de reinício (que olha quem está junto da bola parada: o
cobrador, o goleiro) já decide de quem será a posse seguinte.

  VIVA_M   em jogo, mandante com a bola       -> H
  VIVA_V   em jogo, visitante com a bola      -> A
  MORTA_M  parada, mandante vai repor         -> D
  MORTA_V  parada, visitante vai repor        -> D
"""

import numpy as np

from .config import A, D, FPS, H, X

VIVA_M, VIVA_V, MORTA_M, MORTA_V = 0, 1, 2, 3
ROTULO = np.array([H, A, D, D], dtype=np.int8)
PROIBIDO = 1e9

# custos padrão; o treinar.py escolhe os do modelos/decodificador.json pela validação na DFL
CUSTOS = {"troca": 9.0, "parada": 12.0, "reinicio": 1.0}
VIVA_APOS_KICKOFF = FPS      # frames: o pontapé inicial põe a bola em jogo


def matriz_custos(troca, parada, reinicio):
    c = np.full((4, 4), PROIBIDO)
    np.fill_diagonal(c, 0.0)
    c[VIVA_M, VIVA_V] = c[VIVA_V, VIVA_M] = troca
    c[[VIVA_M, VIVA_M, VIVA_V, VIVA_V], [MORTA_M, MORTA_V, MORTA_M, MORTA_V]] = parada
    c[MORTA_M, VIVA_M] = c[MORTA_V, VIVA_V] = parada
    c[MORTA_M, MORTA_V] = c[MORTA_V, MORTA_M] = reinicio
    return c


def caminho(pontos, custo):
    """Sequência de estados de maior pontuação.

    pontos (n, k): log-pontuação de cada estado em cada frame (maior é melhor).
    custo (k, k) ou (c, k, k): preço de ir do estado i ao j; com c matrizes, decodifica as c de
    uma vez e devolve (c, n). Em empate, fica no estado em que está.
    """
    custo = np.asarray(custo, dtype=np.float64)
    um = custo.ndim == 2
    custo = custo[None] if um else custo
    pontos = np.asarray(pontos, dtype=np.float64)
    n, k = pontos.shape
    c = len(custo)
    volta = np.empty((n, c, k), dtype=np.int8)
    melhor = np.repeat(pontos[:1], c, axis=0)            # (c, k)
    diag = np.arange(k)
    for t in range(1, n):
        cand = melhor[:, :, None] - custo                # (c, de, para)
        de = cand.argmax(axis=1)                         # (c, para)
        valor = np.take_along_axis(cand, de[:, None, :], axis=1)[:, 0, :]
        fica = cand[:, diag, diag] >= valor
        de = np.where(fica, diag[None, :], de)
        volta[t] = de
        melhor = np.where(fica, cand[:, diag, diag], valor) + pontos[t]
    out = np.empty((c, n), dtype=np.int8)
    out[:, -1] = melhor.argmax(axis=1)
    linhas = np.arange(c)
    for t in range(n - 1, 0, -1):
        out[:, t - 1] = volta[t, linhas, out[:, t]]
    return out[0] if um else out


def emissoes(p_morta, p_mandante, p_reinicio, extra_morta=None):
    """(n, 4) log-pontuações. extra_morta soma no log-odds de bola morta: é por onde entra a
    evidência que a DFL não tem (o plano da transmissão)."""
    z = np.log(p_morta) - np.log1p(-p_morta)
    if extra_morta is not None:
        z = z + extra_morta
    log_morta = -np.logaddexp(0.0, -z)
    log_viva = -np.logaddexp(0.0, z)
    s = np.empty((len(z), 4))
    s[:, VIVA_M] = log_viva + np.log(p_mandante)
    s[:, VIVA_V] = log_viva + np.log1p(-p_mandante)
    s[:, MORTA_M] = log_morta + np.log(p_reinicio)
    s[:, MORTA_V] = log_morta + np.log1p(-p_reinicio)
    return s


def _forcar(bloco, forca):
    """forca por frame: 1 = só estados de bola morta, 0 = só de bola viva, -1 = livre."""
    bloco = bloco.copy()
    bloco[np.ix_(forca == 1, [VIVA_M, VIVA_V])] = -PROIBIDO
    bloco[np.ix_(forca == 0, [MORTA_M, MORTA_V])] = -PROIBIDO
    return bloco


def _travar_kickoff(bloco, k, original=None):
    """Antes do kickoff a bola está morta; no primeiro segundo depois, viva. Com `original` (os
    pontos antes de forçar paradas), o kickoff vale mais que a parada forçada."""
    bloco = bloco.copy()
    fim = k + VIVA_APOS_KICKOFF
    if original is not None:
        bloco[:k, [MORTA_M, MORTA_V]] = original[:k, [MORTA_M, MORTA_V]]
        bloco[k:fim, [VIVA_M, VIVA_V]] = original[k:fim, [VIVA_M, VIVA_V]]
    bloco[:k, [VIVA_M, VIVA_V]] = -PROIBIDO
    bloco[k:fim, [MORTA_M, MORTA_V]] = -PROIBIDO
    return bloco


def decodificar(pontos, tempos, custos=None, kickoffs=None, forcar=None):
    """Estados por frame (-1 fora dos tempos). custos: dict como CUSTOS, ou lista de dicts (grade);
    com lista, devolve (len(lista), n). forcar (n,): 1 bola morta, 0 bola viva, -1 o modelo decide
    (a hierarquia rígida: a bola morta vem de fora, o modelo só decide de quem é a bola)."""
    grade = isinstance(custos, (list, tuple))
    lista = custos if grade else [dict(CUSTOS, **(custos or {}))]
    mats = np.stack([matriz_custos(c["troca"], c["parada"], c["reinicio"]) for c in lista])
    out = np.full((len(lista), len(pontos)), -1, dtype=np.int8)
    for tempo, a, b in tempos:
        bloco = original = pontos[a:b + 1]
        if forcar is not None:
            bloco = _forcar(bloco, forcar[a:b + 1])
        k = (kickoffs or {}).get(tempo)
        if k is not None and a <= k <= b:
            bloco = _travar_kickoff(bloco, k - a, original if forcar is not None else None)
        out[:, a:b + 1] = caminho(bloco, mats)
    return out if grade else out[0]


def rotulos(estados):
    """Estados -> códigos H/A/D, X fora dos tempos."""
    out = np.full(len(estados), X, dtype=np.int8)
    ok = estados >= 0
    out[ok] = ROTULO[estados[ok]]
    return out


def adiantar_paradas(rot, tempos, inicio, fim, kickoffs=None):
    """Antecipa as paradas previstas: o começo de cada uma recua `inicio` frames e o fim recua `fim`.

    O calibrar.py mede quanto. Limites, para a correção não apagar nada:
      - o começo avança sobre a posse anterior até a metade dela, nunca mais;
      - o fim só recua se a parada continuar com pelo menos 1 s;
      - a parada que acaba no kickoff fica com o fim onde está (o kickoff é conhecido).
    """
    out = rot.copy()
    ko = set((kickoffs or {}).values())
    for _, a, b in tempos:
        r = rot[a:b + 1]
        cortes = np.flatnonzero(np.diff(r) != 0) + 1
        ini = np.concatenate(([0], cortes))
        fins = np.concatenate((cortes, [len(r)]))
        for i, (s, e) in enumerate(zip(ini, fins)):
            if r[s] != D:
                continue
            if i > 0 and r[ini[i - 1]] != D and inicio > 0:
                antes = (fins[i - 1] - ini[i - 1]) // 2
                passo = min(inicio, antes)
                out[a + s - passo:a + s] = D
            depois = i + 1 < len(ini)
            if fim > 0 and depois and (a + e) not in ko and e - s - fim >= FPS:
                out[a + e - fim:a + e] = r[ini[i + 1]]
    return out
