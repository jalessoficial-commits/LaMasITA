"""Jogos abertos da Bundesliga (Sportec Solutions / DFL, temporada 2022/23) no formato do banco de teste.

    from dfl import carregar_jogo_dfl
    J = carregar_jogo_dfl("J03WN1")

Os arquivos ficam fora do repositório (~350-420 MB de posições por jogo). Baixe do Hugging Face:

    B=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
    curl -L -o mi_J03WN1.xml  $B/DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml
    curl -L -o pos_J03WN1.xml $B/DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml

e aponte DFL_DIR para a pasta (padrão: ../dfl-idsse-data, ao lado do repositório). Fonte e licença:
Bassek et al. (2025), "An integrated dataset of spatiotemporal and event data in elite soccer",
Scientific Data; espelho em huggingface.co/datasets/pysport/idsse-data.

Por que estes jogos: o tracking da DFL traz, em cada frame da bola, BallStatus (1 = em jogo, 0 = morta)
e BallPossession (1 = mandante, 2 = visitante). É um gabarito POR FRAME, bem mais parecido com o do
desafio do que o que validacao_metrica/metrica.py monta a partir de eventos. Como o desafio, ele não
troca a posse em todo toque do adversário. São outros jogos, então podem ser usados para validar
(RULES.md), e precisam ser citados no documento técnico.

Gabarito: D onde BallStatus = 0; H / A pelo BallPossession onde a bola está em jogo; X fora dos tempos.
Coordenadas: a DFL usa a origem no centro do campo; aqui somamos meio campo para ficar como o resto do
banco (0..105 x 0..68 m).
"""

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DFL_DIR = os.environ.get("DFL_DIR", os.path.join(os.path.dirname(ROOT), "dfl-idsse-data"))
FPS = 25

_FS = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)" PersonId="([^"]+)"')
_FR = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"')
_BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


class Jogo:
    pass


def _times(mi_path):
    txt = open(mi_path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"<Team [^>]*>", txt):
        tag = m.group(0)
        role = re.search(r'Role="(\w+)"', tag).group(1)
        tid = re.search(r'TeamId="([^"]+)"', tag).group(1)
        size = re.search(r'PitchSize|', txt)
        out[tid] = "H" if role == "home" else "A"
    x = re.search(r'PitchX="([\d.]+)"', txt)
    y = re.search(r'PitchY="([\d.]+)"', txt)
    return out, (float(x.group(1)) if x else 105.0, float(y.group(1)) if y else 68.0)


def _ler_posicoes(pos_path):
    """{PersonId: (team, frames, x, y)}, bola com posse e status, e os frames de cada tempo."""
    pessoas, bola = {}, None
    cur, buf = None, []
    secoes = {}

    def fechar():
        if cur is None or not buf:
            return
        sec, team, pid = cur
        a = np.array(buf, dtype=np.float64)
        n = a[:, 0].astype(np.int64)
        lo, hi = secoes.get(sec, (n.min(), n.max()))
        secoes[sec] = (min(lo, n.min()), max(hi, n.max()))
        if team == "BALL":
            nonlocal bola
            parte = (n, a[:, 1], a[:, 2], a[:, 3], a[:, 4])
            bola = parte if bola is None else tuple(np.concatenate([u, v]) for u, v in zip(bola, parte))
        elif team != "referee":
            old = pessoas.get(pid)
            parte = (n, a[:, 1], a[:, 2])
            pessoas[pid] = (team, parte) if old is None else (team, tuple(np.concatenate([u, v]) for u, v in zip(old[1], parte)))

    with open(pos_path, encoding="utf-8") as h:
        for line in h:
            if line.startswith("<Frame "):
                m = _FR.match(line)
                if cur is not None and cur[1] == "BALL":
                    b = _BALL.search(line)
                    buf.append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                                int(b.group(1)), int(b.group(2))))
                else:
                    buf.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            elif line.startswith("<FrameSet "):
                fechar()
                f = _FS.match(line)
                cur, buf = (f.group(1), f.group(2), f.group(3)), []
        fechar()
    return pessoas, bola, secoes


def carregar_jogo_dfl(match, base=None, cache=True):
    """Jogo com N, players (n, k, 2) m, time_slot (k,), ball (n, 2) m, gt, reinicios, como metrica.py."""
    base = base or DFL_DIR
    npz = os.path.join(base, "cache_{}.npz".format(match))
    if cache and os.path.exists(npz):
        z = np.load(npz, allow_pickle=False)
        J = Jogo()
        J.players, J.time_slot, J.ball, J.gt = z["players"], z["time_slot"], z["ball"], z["gt"]
        J.N = len(J.gt)
        J.reinicios = _reinicios(J.gt)
        return J
    times, (L, W) = _times(os.path.join(base, "mi_{}.xml".format(match)))
    pessoas, bola, secoes = _ler_posicoes(os.path.join(base, "pos_{}.xml".format(match)))
    n0 = min(lo for lo, _ in secoes.values())
    N = max(hi for _, hi in secoes.values()) - n0 + 1
    ids = sorted(pessoas)
    players = np.full((N, len(ids), 2), np.nan, np.float32)
    time_slot = np.array([times.get(pessoas[p][0], "?") for p in ids])
    for k, p in enumerate(ids):
        n, x, y = pessoas[p][1]
        players[n - n0, k, 0] = x + L / 2
        players[n - n0, k, 1] = y + W / 2
    ball = np.full((N, 2), np.nan, np.float32)
    n, x, y, poss, stat = bola
    ball[n - n0, 0] = x + L / 2
    ball[n - n0, 1] = y + W / 2
    gt = np.full(N, "X", dtype="<U1")
    gt[n - n0] = np.where(stat == 0, "D", np.where(poss == 1, "H", "A"))
    J = Jogo()
    J.N, J.players, J.time_slot, J.ball, J.gt = N, players, time_slot, ball, gt
    J.reinicios = _reinicios(gt)
    if cache:
        np.savez(npz, players=players, time_slot=time_slot, ball=ball, gt=gt)
    return J


def _reinicios(gt):
    """Trechos D dentro dos tempos: inicio, fim (primeiro frame vivo), dono = time no reinício."""
    rows = []
    padded = np.concatenate((["X"], gt, ["X"]))
    d = padded == "D"
    starts = np.flatnonzero(~d[:-1] & d[1:])
    ends = np.flatnonzero(d[:-1] & ~d[1:])
    for s, e in zip(starts, ends):
        if e < len(gt) and gt[e] in ("H", "A") and s > 0 and gt[s - 1] != "X":
            rows.append(dict(inicio=int(s), fim=int(e), dono=gt[e], tipo=""))
    return pd.DataFrame(rows)
