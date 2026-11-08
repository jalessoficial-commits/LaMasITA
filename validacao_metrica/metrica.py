"""Jogos de amostra da Metrica (outros jogos, dados abertos) no formato que a análise usa.

    from metrica import carregar_jogo
    J = carregar_jogo(1)          # 1 ou 2

Os dados ficam fora do repositório. Baixe uma vez:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/metrica-sports/sample-data
    cd sample-data && git sparse-checkout set data/Sample_Game_1 data/Sample_Game_2

e aponte METRICA_DIR para a pasta sample-data (padrão: ../metrica-sample-data, ao lado do repositório).

Por que estes jogos: o RULES.md proíbe rótulos manuais sobre o jogo do desafio, inclusive para validar,
e proíbe usar os 5 minutos públicos para escolher entre versões. Dados rotulados de OUTROS jogos são
permitidos. Aqui os rótulos vêm dos eventos da Metrica (feitos por pessoas) e precisam ser citados no
documento técnico.

Gabarito montado a partir dos eventos (aproximação da definição do CHALLENGE.md):

    reinício      SET PIECE (qualquer subtipo) ou PASS/GOAL KICK; o time do evento é quem cobra
    início do D   o último gatilho entre o reinício anterior e este: BALL OUT e SHOT ...OUT/GOAL no
                  End Frame (a bola cruzando a linha); FAULT RECEIVED, CHALLENGE ...FAULT... no Start
                  Frame (o apito); BALL LOST OFFSIDE/HAND BALL no End Frame
    posse         time do último PASS, SET PIECE, RECOVERY, SHOT ou BALL LOST. Duelos (CHALLENGE) e
                  BALL OUT não trocam a posse: toque sem controle não é troca (CHALLENGE.md)
    X             antes do primeiro reinício e depois do último evento de cada tempo

Tudo indexado do frame 0 (Frame da Metrica - 1), em metros (105 x 68), a 25 fps.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICA_DIR = os.environ.get("METRICA_DIR", os.path.join(os.path.dirname(ROOT), "metrica-sample-data"))
FPS = 25
LENGTH, WIDTH = 105.0, 68.0

EVENTOS_POSSE = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST"}


class Jogo:
    """players (n, k, 2) m · time_slot (k,) 'H'/'A' · ball (n, 2) m (NaN = sem bola) · period (n,)
    gt (n,) 'H'/'A'/'D'/'X' · reinicios DataFrame [inicio, fim, dono, tipo, subtipo]."""


def _tracking(path):
    head = pd.read_csv(path, nrows=2, header=None)
    df = pd.read_csv(path, skiprows=2)
    cols = list(df.columns)
    team = head.iloc[0, 3]                          # "Home" ou "Away"
    xs = [i for i, c in enumerate(cols) if str(c).startswith("Player")]
    xy = np.stack([df.iloc[:, [i, i + 1]].values for i in xs], axis=1).astype(np.float32)
    bi = cols.index("Ball")
    ball = df.iloc[:, [bi, bi + 1]].values.astype(np.float32)
    xy[..., 0] *= LENGTH; xy[..., 1] *= WIDTH
    ball[:, 0] *= LENGTH; ball[:, 1] *= WIDTH
    return df["Period"].values, df["Frame"].values, xy, ball, "H" if team.lower().startswith("home") else "A"


def _gatilho(ev):
    """Frame (base 0) em que este evento para o jogo, ou None."""
    t, s = ev["Type"], str(ev["Subtype"]).upper()
    if t == "BALL OUT":
        return int(ev["End Frame"]) - 1
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return int(ev["End Frame"]) - 1
    if t == "FAULT RECEIVED" or (t == "CHALLENGE" and "FAULT" in s):
        return int(ev["Start Frame"]) - 1
    if t == "BALL LOST" and ("OFFSIDE" in s or "HAND BALL" in s):
        return int(ev["End Frame"]) - 1
    return None


def _tipo(evs):
    """Tipo da parada pelos gatilhos: 'saida' (bola fora), 'gol', 'apito' (falta, impedimento) ou ''."""
    tipos = [(_gatilho(e), e) for _, e in evs.iterrows()]
    tipos = [(f, e) for f, e in tipos if f is not None]
    if not tipos:
        return "", None
    f, e = max(tipos, key=lambda p: p[0])
    s = str(e["Subtype"]).upper()
    if e["Type"] == "SHOT" and "GOAL" in s and "WOODWORK" not in s:
        return "gol", f
    if e["Type"] in ("BALL OUT", "SHOT"):
        return "saida", f
    return "apito", f


def carregar_jogo(n, base=None):
    base = base or METRICA_DIR
    pasta = os.path.join(base, "data", "Sample_Game_{}".format(n))
    nome = os.path.join(pasta, "Sample_Game_{}_".format(n))
    per_h, fr_h, xy_h, ball, _ = _tracking(nome + "RawTrackingData_Home_Team.csv")
    per_a, fr_a, xy_a, _, _ = _tracking(nome + "RawTrackingData_Away_Team.csv")
    assert np.array_equal(fr_h, fr_a) and np.array_equal(fr_h, np.arange(1, len(fr_h) + 1))
    N = len(fr_h)
    J = Jogo()
    J.N, J.period = N, per_h
    J.players = np.concatenate([xy_h, xy_a], axis=1)
    J.time_slot = np.array(["H"] * xy_h.shape[1] + ["A"] * xy_a.shape[1])
    J.ball = ball

    ev = pd.read_csv(nome + "RawEventsData.csv")
    ev["Team"] = ev["Team"].map({"Home": "H", "Away": "A"})
    ev["Subtype"] = ev["Subtype"].fillna("")
    ev = ev.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)
    eh_reinicio = (ev.Type == "SET PIECE") | ((ev.Type == "PASS") & (ev.Subtype.str.upper() == "GOAL KICK"))

    gt = np.full(N, "X", dtype="<U1")
    rows = []
    for p in sorted(set(per_h)):
        e = ev[ev.Period == p]
        r_idx = e.index[eh_reinicio[e.index]].tolist()
        fim_tempo = int(e["End Frame"].max())                     # depois disso: X
        pos = e[e.Type.isin(EVENTOS_POSSE)]
        # posse por frame: time do último evento de posse (forward fill), só dentro do tempo
        lab = np.full(N, "", dtype="<U1")
        lab[(pos["Start Frame"].values - 1).astype(int)] = pos["Team"].values
        ini = int(e.loc[r_idx[0], "Start Frame"]) - 1
        seg = lab[ini:fim_tempo]
        idx = np.where(seg != "", np.arange(len(seg)), 0)
        np.maximum.accumulate(idx, out=idx)
        gt[ini:fim_tempo] = seg[idx]
        # paradas: [último gatilho, reinício)
        for k, i in enumerate(r_idx):
            fim = int(ev.loc[i, "Start Frame"]) - 1
            if k == 0:
                continue                                          # pontapé inicial do tempo
            entre = ev.loc[r_idx[k - 1] + 1:i - 1]
            tipo, inicio = _tipo(entre)
            if inicio is None or inicio >= fim:                   # parada sem gatilho nos eventos
                antes = ev.loc[r_idx[k - 1]:i - 1]
                inicio = int(min(antes["End Frame"].max(), fim)) - 1 if len(antes) else fim - FPS
                tipo = tipo or "sem_gatilho"
            inicio = max(inicio, int(ev.loc[r_idx[k - 1], "Start Frame"]))
            gt[inicio:fim] = "D"
            gt[fim] = ev.loc[i, "Team"]                           # o reinício é do time que cobra
            rows.append(dict(inicio=inicio, fim=fim, dono=ev.loc[i, "Team"], tipo=tipo,
                             subtipo=ev.loc[i, "Subtype"] or ev.loc[i, "Type"], periodo=p))
    J.gt = gt
    J.reinicios = pd.DataFrame(rows)
    return J
