r"""Detector de paradas v2 (M4): máquina de estados determinística sobre eventos.

    from paradas import Config, detectar_paradas
    dead, tabela = detectar_paradas(S, Config())

Cada evento de início (apito, saída da bola) abre uma parada CANDIDATA. Ela só vira MORTA se
uma CONFIRMAÇÃO aparecer -- e a confirmação, o tempo máximo e o reinício dependem do TIPO:

    VIVA --(início)--> CANDIDATA --(confirmação)--> MORTA --(reinício)--> VIVA
                           \--(nada confirma: vantagem, apito da torcida)--> VIVA

confirmações (qualquer uma)
    v1           o detector v1 (velocidade dos próximos 5 s < limiar) dispara dentro da janela do tipo
    tag          o Smart Tagging tem um reinício desse tipo dentro do tempo máximo
    area         (só linha de fundo) os jogadores convergem para as áreas: +3 ou mais em 15 s
    freio        (só apito) a velocidade média (1 s) cai à metade em até 4 s: o jogo parou no apito.
                 Um apito sem freio é vantagem, apito da torcida ou de outro campo -- descartado.
reinício (o primeiro que valer)
    tag          start_frame + deslocamento típico do tipo (eventos.calibrar_offset_smart_tagging)
    v1           fim do trecho v1 que confirmou a parada: a velocidade futura volta a subir
    aceleracao   mínimo de velocidade seguido de arranque coletivo
    retomada     (só apito com freio) a velocidade volta a subir acima do mínimo + 0,5 m/s
guarda de jogo vivo
    entre o início e o reinício o bloco não pode ter um trecho de > GUARD_S com velocidade média (1 s)
    > GUARD_SPEED -- isso é jogo corrido, não uma parada.

Os trechos v1 que nenhum evento explica continuam como D (ramo "sem evento", precisão ~0,9 no GT).
Saída: máscara D por frame e uma tabela com [inicio, fim, tipo, fonte_inicio, fonte_fim, confirmacao].
"""

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import arvore as T  # noqa: E402
import eventos as E  # noqa: E402
import rotulos as R  # noqa: E402

FPS = 25


@dataclass
class Config:
    usar_inicio: tuple = ("apito", "saida_bola")
    usar_reinicio: tuple = ("smart_tagging", "v1", "aceleracao")
    usar_v1_sem_evento: bool = True
    v1_uniao: bool = False               # True: um evento dentro de um trecho v1 não corta o trecho
    ramos_por_tipo: bool = True
    v1_limiar: float = 1.1               # speed_fut5s_mean < limiar (notebook 02, ajustado no fraco)
    v1_min_dead_s: float = 3.0
    v1_min_alive_s: float = 2.0
    guard_speed: float = 2.2             # m/s (média de 1 s) ...
    guard_s: float = 4.0                 # ... sustentada por mais de 4 s = jogo corrido
    apito_reinicio_s: float = 3.0        # apito entre [start_frame - 3 s, reinício + 3 s] de uma tag autoriza o
                                         # reinício (saída de bola, cobrança): não abre parada
    freio_s: float = 4.0                 # apito: a velocidade precisa cair ...
    freio_frac: float = 0.5              # ... a esta fração do pico em volta do apito
    retomada_ganho: float = 0.5          # m/s acima do mínimo ...
    retomada_min: float = 1.6            # ... e acima disto = reinício
    apito_min_snr: float = 10.0
    fundir_s: float = 3.0                # funde dois D separados por menos de 3 s de "viva"
    # por tipo: (janela de confirmação v1 [s], tempo máximo até o reinício [s], antecipação [s])
    tipos: dict = field(default_factory=lambda: {
        "apito":          (8, 90, 0.3),            # 90 s: a comemoração de gol até a saída de bola
        "linha_de_fundo": (20, 60, 0.0),
        "lateral":        (12, 40, 0.0),
        "centro":         (15, 90, 0.0),
        "campo":          (8, 40, 0.0),
        "desconhecida":   (8, 40, 0.0),
    })
    tipos_tag: dict = field(default_factory=lambda: {   # que tags fecham que tipo de início
        "apito": None,                                  # None = qualquer tag
        "linha_de_fundo": ("Corner Kick", "Goal Kick"),
        "lateral": ("Throw-in",),
        "centro": ("Kick Off (After Goal)", "Kick Off (Start)"),
        "campo": ("Free Kick",),
        "desconhecida": None,
    })


def v1_runs(S, cfg):
    """D da árvore v1 (notebook 02: nós N1+N2 e limpeza), lido de out/arvore_D.csv; se não existir,
    só o nó N1."""
    path = os.path.join(HERE, "out", "arvore_D.csv")
    if os.path.exists(path):
        dead = pd.read_csv(path, usecols=["morta"]).morta.values.astype(bool)
    else:
        raw = S.f("speed_fut5s_mean") < cfg.v1_limiar
        dead = T.clean_runs(raw & S.graded, cfg.v1_min_dead_s, cfg.v1_min_alive_s)
    dead = dead & S.graded
    return dead, T.runs(dead)


def _guard_ok(S, a, b, cfg):
    """True se não houver jogo corrido (velocidade alta sustentada) dentro de [a, b)."""
    if b - a <= 0:
        return True
    fast = S.speed1[a:b] > cfg.guard_speed
    return not any(e - s > cfg.guard_s * FPS for s, e in R.runs(fast))


def _freio(S, f0, cfg):
    s1 = S.speed1
    pico = np.nanmax(s1[max(0, f0 - FPS):f0 + FPS])
    depois = np.nanmin(s1[f0:f0 + int(cfg.freio_s * FPS)])
    return bool(depois <= cfg.freio_frac * pico or depois < 1.0)


def _retomada(S, a, b, cfg):
    """Primeiro frame em [a, b) em que a velocidade sobe acima do mínimo corrente + ganho."""
    s1 = S.speed1[a:b]
    if not len(s1):
        return None
    run_min = np.minimum.accumulate(np.nan_to_num(s1, nan=9.9))
    k = np.flatnonzero((s1 > run_min + cfg.retomada_ganho) & (s1 > cfg.retomada_min))
    return int(a + k[0]) if len(k) else None


def detectar_paradas(S, cfg=None, inicio=None, reinicio=None, offsets=None, v1_dead=None):
    """v1_dead: máscara D da árvore v1; None = a do notebook 02 (out/arvore_D.csv)."""
    cfg = cfg or Config()
    N = S.N
    if v1_dead is None:
        v1_dead, v1r = v1_runs(S, cfg)
    else:
        v1_dead = v1_dead & S.graded
        v1r = T.runs(v1_dead)
    v1_starts = np.array([s for s, _ in v1r]); v1_ends = np.array([e for _, e in v1r])
    inicio = inicio if inicio is not None else E.eventos_inicio(S, cfg.usar_inicio)
    inicio = [e for e in inicio if e["fonte"] != "apito" or e["forca"] >= cfg.apito_min_snr]
    if offsets is None:
        offsets, _ = E.calibrar_offset_smart_tagging(S)
    tags = E.eventos_smart_tagging(S, offsets) if "smart_tagging" in cfg.usar_reinicio else []
    acel = E.eventos_aceleracao(S) if "aceleracao" in cfg.usar_reinicio else []
    tag_f = np.array([t["frame"] for t in tags]); tag_t = np.array([t["tipo"] for t in tags])
    if len(tag_f):   # apitos que autorizam um reinício marcado não abrem parada
        tag_sf = np.array([t["start_frame"] for t in tags])
        w = cfg.apito_reinicio_s * FPS
        inicio = [e for e in inicio if e["fonte"] != "apito"
                  or not np.any((e["frame"] >= tag_sf - w) & (e["frame"] <= tag_f + w))]
    acel_f = np.array([a["frame"] for a in acel])
    boxes = S.f("all_players_in_boxes")

    rows = []
    dead = np.zeros(N, bool)
    busy_until = -1
    for ev in inicio:
        f0 = ev["frame"]
        if f0 < busy_until:
            continue                                   # já dentro de uma parada aberta
        tipo = ev["tipo"] if cfg.ramos_por_tipo else "desconhecida"
        if ev["fonte"] == "apito":
            tipo = "apito"
        conf_s, max_s, antecipa = cfg.tipos.get(tipo, cfg.tipos["desconhecida"])
        start = max(0, f0 - int(antecipa * FPS))
        horizon = min(N - 1, f0 + int(max_s * FPS))
        confirm, end, fonte_fim = None, None, None

        # (1) o PRIMEIRO reinício marcado pelo Smart Tagging (qualquer tipo: houve reinício)
        tag_ok = None
        if len(tag_f):
            k = np.flatnonzero((tag_f > f0 + 2 * FPS) & (tag_f <= horizon))
            if len(k) and _guard_ok(S, f0 + 5 * FPS, tag_f[k[0]] - 3 * FPS, cfg):
                confirm, end, fonte_fim = "tag", int(tag_f[k[0]]), "smart_tagging"
                ok_types = cfg.tipos_tag.get(tipo)
                tag_ok = ok_types is None or tag_t[k[0]] in ok_types

        # (2) trecho v1 que começa dentro da janela de confirmação
        if end is None and "v1" in cfg.usar_reinicio and len(v1_starts):
            k = np.flatnonzero((v1_ends > f0) & (v1_starts <= f0 + conf_s * FPS))
            if len(k) and _guard_ok(S, f0 + 5 * FPS, v1_starts[k[0]], cfg):
                confirm, end, fonte_fim = "v1", int(v1_ends[k[0]]), "v1"

        # (3) linha de fundo: convergência para as áreas (escanteio), fecha na próxima aceleração
        if end is None and cfg.ramos_por_tipo and tipo == "linha_de_fundo":
            w = boxes[f0:min(N, f0 + 15 * FPS)]
            if len(w) and np.nanmax(w) - np.nanmin(w[:FPS]) >= 3:
                k = acel_f[(acel_f > f0 + 5 * FPS) & (acel_f <= horizon)] if len(acel_f) else []
                if len(k) and _guard_ok(S, f0 + 5 * FPS, k[0], cfg):
                    confirm, end, fonte_fim = "area", int(k[0]), "aceleracao"

        # (4) apito: freio coletivo logo depois do apito; fecha na retomada da velocidade
        freio = tipo == "apito" and _freio(S, f0, cfg)
        if tipo == "apito" and cfg.ramos_por_tipo:
            if not freio and confirm != "tag":
                continue                               # apito sem freio nem reinício marcado: vantagem ou torcida
            if end is None:
                end = _retomada(S, f0 + int(cfg.freio_s * FPS), horizon, cfg)
                if end is not None:
                    confirm, fonte_fim = "freio", "retomada"
        # tag de reinício incompatível com o tipo do início: o início não é desta parada
        if cfg.ramos_por_tipo and confirm == "tag" and tag_ok is False:
            continue
        if end is None:
            continue
        # um reinício por aceleração logo antes do fim proposto corta a parada mais cedo
        if "aceleracao" in cfg.usar_reinicio and fonte_fim == "v1" and len(acel_f):
            k = acel_f[(acel_f > f0 + 3 * FPS) & (acel_f < end - 2 * FPS)]
            if len(k):
                end, fonte_fim = int(k[0]), "aceleracao"
        dead[start:end] = True
        busy_until = end
        rows.append(dict(inicio=start, fim=end, dur_s=(end - start) / FPS, tipo=tipo, fonte_inicio=ev["fonte"],
                         fonte_fim=fonte_fim, confirmacao=confirm, forca=round(ev["forca"], 1),
                         tag_compativel=tag_ok))

    # ramo "sem evento": trechos v1 que nenhuma parada cobre
    if cfg.usar_v1_sem_evento:
        for s, e in v1r:
            if not dead[s:e].any():
                dead[s:e] = True
                rows.append(dict(inicio=s, fim=e, dur_s=(e - s) / FPS, tipo="sem_evento", fonte_inicio="v1",
                                 fonte_fim="v1", confirmacao="v1", forca=np.nan))
            elif cfg.v1_uniao:
                dead[s:e] = True
    # buracos curtos de viva entre dois D (o v1 às vezes parte uma parada lenta em duas)
    for s_, e_ in T.runs(~dead):
        if 0 < s_ and e_ < N and e_ - s_ < cfg.fundir_s * FPS:
            dead[s_:e_] = True
    dead &= S.graded
    tabela = pd.DataFrame(rows).sort_values("inicio").reset_index(drop=True) if rows else pd.DataFrame()
    return dead, tabela


def exportar_viewer(tabela, S, path=None):
    """viewer/data/eventos.json: apitos (tiques) e paradas (barras por tipo)."""
    path = path or os.path.join(ROOT, "viewer", "data", "eventos.json")
    apitos = [[int(a), int(b), float(s)] for a, b, s in S.apito_meta.get("eventos", [])]
    paradas = [dict(start_frame=int(r.inicio), end_frame=int(r.fim), tipo=r.tipo,
                    fonte_inicio=r.fonte_inicio, fonte_fim=r.fonte_fim) for r in tabela.itertuples()]
    with open(path, "w", encoding="utf-8") as h:
        json.dump({"apitos": apitos, "paradas": paradas}, h)
    return path
