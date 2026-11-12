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
    # ---- v3 (desligado por padrão: Config() continua sendo o v2) --------------------------------
    gols: bool = False                   # ramo gol: GOALS do Smart Tagging -> D do gol até o kick-off
    gol_ausencia_frac: float = 0.85      # depois do gol a bola fica sumida em >= 85% dos frames até o kick-off
    gol_visivel_s: float = 5.0           # trecho de bola visível que conta como "o lance"
    gol_apito_s: float = 20.0            # um apito até 20 s antes do sumiço marca o início
    gol_anulado_s: float = 150.0         # GOALS sem Kick Off (After Goal) em 150 s: anulado, fecha na tag
    veto_bola_viva: bool = False         # parada sem tag com a bola visível e longe das linhas = jogo
    veto_pres: float = 0.9               # ... bola presente em >= 90% dos frames
    veto_linha_m: float = 2.0            # ... e nunca a menos de 2 m de uma linha
    pausa_fundir_s: float = 0.0          # pausa técnica: funde D separados por até N s de jogo lento
    pausa_vel: float = 1.5               # ... velocidade média no buraco < 1,5 m/s (VAR: 1,22)
    pausa_min_s: float = 30.0            # ... e um dos lados com >= 30 s
    inicio_de_tempo: bool = False        # D do começo do tempo até o Kick Off (Start)
    # ---- v4 (vereditos dos clipes do 1º tempo, 2026-11-12) ---------------------------------------
    retro_apito_s: float = 0.0           # trecho v1 sem evento: recua o início até um apito forte descartado
    retro_sumico_s: float = 0.0          # ... ou até onde a bola sumiu com o jogo já lento (< retro_vel)
    retro_vel: float = 2.0
    viva_funda_m: float = 0.0            # bola visível, >= N m dentro do campo e rápida = jogo: corta o D
    viva_funda_vel: float = 4.0          # ... velocidade da bola (m/s, média de 1 s)
    viva_funda_min_s: float = 1.0        # ... por pelo menos N s seguidos
    viva_funda_jog: float = 1.5          # ... com os jogadores em jogo (velocidade média >= N m/s)
    viva_funda_sem_evento: bool = True   # aplica também aos trechos v1 sem evento
    viva_funda_antes_s: float = 6.0      # ... a parte antes do jogo voltar fica se tiver >= N s
    goleiro_m: float = 0.0               # fim de trecho v1 com a bola na mão do goleiro (a <= N m) = jogo

    @classmethod
    def v3(cls, **kw):
        base = dict(v1_uniao=True, gols=True, veto_bola_viva=True, pausa_fundir_s=10.0, inicio_de_tempo=True)
        base.update(kw)
        return cls(**base)

    @classmethod
    def v4(cls, **kw):
        base = dict(retro_apito_s=15.0, retro_sumico_s=20.0, viva_funda_m=12.0, viva_funda_sem_evento=False,
                    goleiro_m=3.5, gol_apito_s=8.0)
        base.update(kw)
        return cls.v3(**base)


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


def _out_dist(S):
    if getattr(S, "_od", None) is None:
        x, y = S.ball[:, 0], S.ball[:, 1]
        S._od = np.maximum.reduce([-x, x - E.LENGTH, -y, y - E.WIDTH])     # >0 fora, <0 dentro
    return S._od


def _vel_bola(S):
    if getattr(S, "_vb", None) is None:
        v = np.full(len(S.ball), np.nan)
        v[1:] = np.hypot(*(S.ball[1:] - S.ball[:-1]).T) * FPS
        v[v > 40] = np.nan                                       # saltos do tracking
        S._vb = pd.Series(v).rolling(FPS, center=True, min_periods=FPS // 3).mean().values
    return S._vb


def _viva_funda(S, cfg):
    """v4: bola vista longe das linhas e rolando rápido por >= 1 s: jogo, não bola parada."""
    m = (_out_dist(S) < -cfg.viva_funda_m) & (_vel_bola(S) > cfg.viva_funda_vel) & (S.speed1 >= cfg.viva_funda_jog)
    out = np.zeros(len(m), bool)
    for s_, e_ in T.runs(m):
        if e_ - s_ >= cfg.viva_funda_min_s * FPS:
            out[s_:e_] = True
    return out


def _goleiro_com_bola(S, cfg):
    """v4: bola parada na área (fora da pequena área) colada no jogador mais recuado daquele lado.
    A pequena área fica de fora porque ali o goleiro com a bola é o tiro de meta (bola morta)."""
    if getattr(S, "_gk", None) is not None:
        return S._gk
    bx, by = S.ball[:, 0], S.ball[:, 1]
    px, py = S.players[:, :, 0], S.players[:, :, 1]
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    near, dnear = np.argmin(d, axis=1), np.min(d, axis=1)
    xmin = np.argmin(np.where(np.isnan(px), np.inf, px), axis=1)
    xmax = np.argmax(np.where(np.isnan(px), -np.inf, px), axis=1)
    na_area = np.abs(by - E.WIDTH / 2) < 21
    esq = na_area & (bx > 5.5) & (bx < 17.5) & (near == xmin)
    dir_ = na_area & (bx < E.LENGTH - 5.5) & (bx > E.LENGTH - 17.5) & (near == xmax)
    S._gk = (dnear <= cfg.goleiro_m) & (esq | dir_) & (_vel_bola(S) < 3)
    return S._gk


def _bola_viva(S, a, b, cfg):
    """v3: a bola foi vista em quase todo o trecho e nunca chegou perto de uma linha -- o jogo seguiu
    (apito da torcida, vantagem, jogo lento). Nos D verdadeiros o ATD perde a bola ou ela sai."""
    if b - a <= 0:
        return False
    pres = np.mean(~np.isnan(S.ball[a:b, 0]))
    od = _out_dist(S)[max(0, a - 2 * FPS):b]
    perto = np.nanmax(od) if np.any(~np.isnan(od)) else -np.inf
    return bool(pres >= cfg.veto_pres and perto < -cfg.veto_linha_m)


def paradas_de_gol(S, cfg, tags, apitos):
    """v3: um D por GOALS do Smart Tagging (o span da tag cobre o lance inteiro, não o gol).

    fim     reinício do Kick Off (After Goal) seguinte; sem ele em gol_anulado_s, a última tag de
            reinício até lá (gol anulado: tiro livre, tiro de meta).
    início  o ATD perde a bola na comemoração: o gol é o FIM do último trecho com a bola visível por
            >= gol_visivel_s antes do kick-off (sem contar os 30 s finais, quando a bola é reposta no
            centro). Um apito até gol_apito_s antes disso (o do gol) vence. Sem trecho visível: o último
            apito da janela."""
    gols = R.smart_tags("GOALS")
    pres = ~np.isnan(S.ball[:, 0])
    out = []
    for g in gols.itertuples():
        g0 = int(g.start_frame)
        lim = g.end_frame + cfg.gol_anulado_s * FPS
        ko = [t for t in tags if t["tipo"] == "Kick Off (After Goal)" and g0 <= t["start_frame"] <= lim]
        if ko:
            fim, fonte_fim = ko[0]["frame"], "kickoff"
        else:
            nxt = [t for t in tags if g0 < t["start_frame"] and t["frame"] <= lim]
            if not nxt:
                continue
            fim, fonte_fim = nxt[-1]["frame"], "tag (anulado)"
        corte = fim - 30 * FPS
        vis = [(g0 + s, g0 + e) for s, e in R.runs(pres[g0:fim]) if g0 + s < corte and e - s >= FPS]
        longos = [r for r in vis if r[1] - r[0] >= cfg.gol_visivel_s * FPS]
        ini, fonte_ini = None, None
        if vis:
            ini, fonte_ini = int((longos or vis)[-1][1]), "bola_sumiu"
            if np.mean(~pres[ini:fim - 3 * FPS]) < cfg.gol_ausencia_frac:
                ini = None
        ap = [a for a in apitos if g0 <= a < fim - 5 * FPS]
        if ini is None:
            if not ap:
                continue
            ini, fonte_ini = ap[-1], "apito"
        else:
            ap = [a for a in ap if ini - cfg.gol_apito_s * FPS <= a <= ini]
            if ap:
                ini, fonte_ini = ap[0], "apito"
        out.append(dict(inicio=int(ini), fim=int(fim), dur_s=(fim - ini) / FPS, tipo="gol", fonte_inicio=fonte_ini,
                        fonte_fim=fonte_fim, confirmacao="tag", forca=np.nan, tag_compativel=True))
    return out


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
        autoriza = [e for e in inicio if e["fonte"] == "apito"
                    and np.any((e["frame"] >= tag_sf - w) & (e["frame"] <= tag_f + w))]
        inicio = [e for e in inicio if not any(e is x for x in autoriza)]
    else:
        autoriza = []
    acel_f = np.array([a["frame"] for a in acel])
    boxes = S.f("all_players_in_boxes")

    rows = []
    descartes = [dict(frame=e["frame"], fonte="apito", tipo="apito", motivo="apito autoriza reinício marcado")
                 for e in autoriza]
    dead = np.zeros(N, bool)
    busy_until = -1
    travado = np.zeros(N, bool)          # v3: D já decidido (gol, início de tempo); eventos ali não abrem nada
    if cfg.gols:
        apitos = [e["frame"] for e in inicio if e["fonte"] == "apito"]    # já sem os que autorizam reinício
        for r in paradas_de_gol(S, cfg, tags, apitos):
            dead[r["inicio"]:r["fim"]] = travado[r["inicio"]:r["fim"]] = True
            rows.append(r)
    if cfg.inicio_de_tempo:
        blocos = [s_ for s_, _ in T.runs(S.graded)]
        for t in tags:
            antes = [b for b in blocos if b <= t["start_frame"]]
            if t["tipo"] != "Kick Off (Start)" or not antes:
                continue
            b0, ko = antes[-1], t["start_frame"]          # o start_frame do Kick Off (Start) já é o pontapé
            if 0 < ko - b0 < 10 * 60 * FPS:
                dead[b0:ko] = travado[b0:ko] = True
                rows.append(dict(inicio=b0, fim=ko, dur_s=(ko - b0) / FPS, tipo="inicio_de_tempo",
                                 fonte_inicio="tempo", fonte_fim="kickoff", confirmacao="tag", forca=np.nan,
                                 tag_compativel=True))

    viva = _viva_funda(S, cfg) if cfg.viva_funda_m > 0 else None
    apitos_sem_freio = []
    tag_sf_all = np.array([t["start_frame"] for t in tags]) if len(tags) else np.array([])

    def descarta(ev, motivo):
        descartes.append(dict(frame=ev["frame"], fonte=ev["fonte"], tipo=ev.get("tipo"), motivo=motivo))

    for ev in inicio:
        f0 = ev["frame"]
        if f0 < busy_until:
            descarta(ev, "dentro de parada aberta")
            continue
        if travado[f0]:
            descarta(ev, "dentro de gol / início de tempo")
            continue
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
                apitos_sem_freio.append(f0)
                descarta(ev, "apito sem freio nem tag")
                continue                               # apito sem freio nem reinício marcado: vantagem ou torcida
            if end is None:
                end = _retomada(S, f0 + int(cfg.freio_s * FPS), horizon, cfg)
                if end is not None:
                    confirm, fonte_fim = "freio", "retomada"
        # tag de reinício incompatível com o tipo do início: o início não é desta parada
        if cfg.ramos_por_tipo and confirm == "tag" and tag_ok is False:
            descarta(ev, "tag incompatível com o tipo")
            continue
        if end is None:
            descarta(ev, "nada confirmou")
            continue
        # um reinício por aceleração logo antes do fim proposto corta a parada mais cedo
        if "aceleracao" in cfg.usar_reinicio and fonte_fim == "v1" and len(acel_f):
            k = acel_f[(acel_f > f0 + 3 * FPS) & (acel_f < end - 2 * FPS)]
            if len(k):
                end, fonte_fim = int(k[0]), "aceleracao"
        if cfg.veto_bola_viva and confirm != "tag" and _bola_viva(S, start, end, cfg):
            descarta(ev, "bola viva (veto v3)")
            continue
        if travado[start:end].any():                   # não invade um gol: para no começo dele
            end = start + int(np.argmax(travado[start:end]))
        partes = [(start, end, ev["fonte"])]
        a0 = max(start, f0) + 2 * FPS                  # a bola ainda rola logo depois do apito / da saída
        if viva is not None and viva[a0:max(a0, end - 3 * FPS)].any():
            # v4: o jogo voltou no meio (bola rápida e longe das linhas). Uma parada real antes disso fica
            # (até 1 s antes); depois, só a partir do próximo evento de início antes do reinício.
            hits = np.flatnonzero(viva[a0:end - 3 * FPS])
            pv, ult = a0 + int(hits[0]), a0 + int(hits[-1]) + 1
            prox = [e for e in inicio if ult <= e["frame"] < end - 2 * FPS]
            partes = []
            if pv - FPS - start >= cfg.viva_funda_antes_s * FPS:
                partes.append((start, pv - FPS, ev["fonte"]))
            if prox:
                partes.append((prox[0]["frame"], end, prox[0]["fonte"]))
            elif confirm == "tag" and not partes:
                partes.append((start, end, ev["fonte"]))   # reinício marcado e nada melhor: fica como está
            if not partes:
                descarta(ev, "bola viva funda (v4)")
                continue
        for k_, (a_, b_, fonte_) in enumerate(partes):
            dead[a_:b_] = True
            rows.append(dict(inicio=a_, fim=b_, dur_s=(b_ - a_) / FPS, tipo=tipo, fonte_inicio=fonte_,
                             fonte_fim=fonte_fim if b_ == end else "bola viva (v4)", confirmacao=confirm,
                             forca=round(ev["forca"], 1), tag_compativel=tag_ok))
        busy_until = end

    # ramo "sem evento": trechos v1 que nenhuma parada cobre
    if cfg.usar_v1_sem_evento:
        for s, e in v1r:
            if not dead[s:e].any():
                if cfg.veto_bola_viva and _bola_viva(S, s, e, cfg):
                    descartes.append(dict(frame=s, fonte="v1", tipo="sem_evento", motivo="bola viva (veto v3)"))
                    continue
                fonte_ini = "v1"
                if viva is not None and cfg.viva_funda_sem_evento and viva[s:e].any():
                    s = s + int(np.flatnonzero(viva[s:e])[-1]) + 1
                if cfg.retro_apito_s > 0:
                    w = [a for a in apitos_sem_freio if s - cfg.retro_apito_s * FPS <= a < s and not dead[a:s].any()]
                    if w:
                        s, fonte_ini = w[0], "apito (recuo v4)"
                # recuo pelo sumiço só com um reinício marcado logo depois (o que confirma a parada)
                marcado = len(tag_f) and np.any((tag_sf_all >= s) & (tag_sf_all <= e + 10 * FPS))
                if fonte_ini == "v1" and cfg.retro_sumico_s > 0 and marcado:
                    a = s
                    lim = max(0, s - int(cfg.retro_sumico_s * FPS))
                    while a > lim and np.isnan(S.ball[a - 1, 0]) and S.speed1[a - 1] < cfg.retro_vel and not dead[a - 1]:
                        a -= 1
                    if a < s:
                        s, fonte_ini = a, "bola_sumiu (recuo v4)"
                if e - s < 2 * FPS:
                    continue
                dead[s:e] = True
                rows.append(dict(inicio=s, fim=e, dur_s=(e - s) / FPS, tipo="sem_evento", fonte_inicio=fonte_ini,
                                 fonte_fim="v1", confirmacao="v1", forca=np.nan))
            elif cfg.v1_uniao:
                dead[s:e] = True
    # v4: goleiro com a bola nas mãos no fim de uma parada fechada pelo v1 = o jogo já voltou
    if cfg.goleiro_m > 0:
        gk = _goleiro_com_bola(S, cfg)
        for r in rows:
            if r["fonte_fim"] != "v1" or r["tipo"] in ("gol", "inicio_de_tempo"):
                continue
            a, b = r["inicio"], r["fim"]
            hits = np.flatnonzero(gk[a:b])
            if len(hits) >= 2 * FPS:
                corte = a + int(hits[0])
                # o que vem depois do goleiro pegar a bola é jogo (se a bola ainda não tinha sumido de novo)
                dead[corte:b] = False
                r["fim"], r["fonte_fim"], r["dur_s"] = corte, "goleiro (v4)", (corte - a) / FPS
                if corte - a < 4 * FPS:
                    dead[a:corte] = False
                    r["descartada"] = True
        rows = [r for r in rows if not r.get("descartada")]
    # buracos curtos de viva entre dois D (o v1 às vezes parte uma parada lenta em duas)
    for s_, e_ in T.runs(~dead):
        if 0 < s_ and e_ < N and e_ - s_ < cfg.fundir_s * FPS:
            dead[s_:e_] = True
    # v3, pausa técnica (VAR, substituição, atendimento): uma parada longa partida pelo v1 em pedaços
    if cfg.pausa_fundir_s > 0:
        d_runs = T.runs(dead)
        for (s1, e1), (s2, e2) in zip(d_runs[:-1], d_runs[1:]):
            longo = max(e1 - s1, e2 - s2) >= cfg.pausa_min_s * FPS
            if longo and s2 - e1 <= cfg.pausa_fundir_s * FPS and np.nanmean(S.speed1[e1:s2]) < cfg.pausa_vel:
                dead[e1:s2] = True
    dead &= S.graded
    tabela = pd.DataFrame(rows).sort_values("inicio").reset_index(drop=True) if rows else pd.DataFrame()
    tabela.attrs["descartes"] = pd.DataFrame(descartes)
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
