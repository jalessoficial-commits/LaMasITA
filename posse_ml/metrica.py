"""Os 3 jogos abertos da Metrica Sports (github.com/metrica-sports/sample-data) como Partida, com
rótulos de posse derivados dos eventos.

    python -m posse_ml.metrica            # resumo dos rótulos de cada jogo, ao lado do GT público

Os jogos trazem rastreamento a 25 Hz (22 jogadores e a bola, coordenadas de 0 a 1) e eventos
anotados (passes, recuperações, perdas, bolas paradas...), mas não o rótulo de posse frame a frame.
Ele sai assim:

  bola morta (D)  a bola dos dados é reconstruída a partir dos eventos e some quando o jogo para:
                  bola ausente por FALTA_MIN frames ou mais é bola morta. Ausências mais curtas
                  continuam com o time que estava com ela.
  posse (H/A)     o time do último evento "com a bola" (passe, condução, finalização, bola parada,
                  recuperação, perda). Uma perda continua com o time que perdeu até o adversário
                  recuperar, como no desafio (a bola no ar segue com o último time).
  toque sem controle  um trecho de um time sem nenhum passe, finalização nem bola parada e mais
                  curto que CONTROLE_MIN frames (2 s) não conta como posse (o desafio pede posse
                  controlada): fica com o time de antes. Com 2 s, a duração das posses fica com a
                  mesma distribuição do GT público (mediana ~7,5 s, 10% abaixo de ~2 s).
  fora do jogo (X)  antes do primeiro evento de cada tempo e depois do último.

Os jogos 1 e 2 vêm em CSV (um arquivo por time); o 3, no formato EPTS/FIFA (metadata.xml + txt), o
mesmo do ATD do desafio. O time "Home" (ou "Team A") é o mandante. Baixar: baixar_metrica.py.
"""

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config
from .config import A, D, H, MANDANTE, VISITANTE, X
from .janelas import trechos
from .partida import Partida

JOGOS = ("MET1", "MET2", "MET3")
CAMPO = (105.0, 68.0)
FALTA_MIN = 25          # frames sem bola para ser bola morta (1 s)
CONTROLE_MIN = 50       # frames (2 s) de um trecho sem passe, chute nem bola parada para contar como posse
COM_BOLA = ("PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST")
CONTROLADA = ("PASS", "SHOT", "SET PIECE")


def pasta_do_jogo(id_jogo, pasta=None):
    return os.path.join(pasta or config.METRICA, "Sample_Game_" + id_jogo[-1])


def jogos(pasta=None):
    """Os jogos da Metrica presentes na pasta."""
    return [j for j in JOGOS if os.path.isdir(pasta_do_jogo(j, pasta))
            and len(os.listdir(pasta_do_jogo(j, pasta))) >= 3]


def eh_metrica(id_jogo):
    return id_jogo in JOGOS


# ------------------------------------------------------------------------------------------------
# leitura
def _ler_csv(pasta, g):
    """Jogos 1 e 2: (jogadores (n, k, 2) em 0-1, lado (k,), bola (n, 2), tempo (n,), eventos)."""
    base = os.path.join(pasta, "Sample_Game_{}_".format(g))
    partes, lados, bola, tempo = [], [], None, None
    for nome, lado in (("Home", MANDANTE), ("Away", VISITANTE)):
        t = pd.read_csv(base + "RawTrackingData_{}_Team.csv".format(nome), skiprows=2)
        v = t.iloc[:, 3:].to_numpy(dtype=np.float32)
        k = (v.shape[1] - 2) // 2
        partes.append(v[:, :2 * k].reshape(len(t), k, 2))
        lados += [lado] * k
        bola, tempo = v[:, 2 * k:2 * k + 2], t["Period"].to_numpy()
    n = min(len(p) for p in partes)
    ev = pd.read_csv(base + "RawEventsData.csv")
    eventos = [(MANDANTE if t == "Home" else VISITANTE, tipo, int(a) - 1, int(b) - 1, int(per))
               for t, tipo, a, b, per in zip(ev["Team"], ev["Type"], ev["Start Frame"], ev["End Frame"], ev["Period"])]
    return np.concatenate([p[:n] for p in partes], axis=1), np.array(lados, dtype=np.int8), bola[:n], tempo[:n], eventos


def _ler_epts(pasta, g):
    """Jogo 3: o txt tem, por frame, 22 colunas de jogador e a bola; o metadata diz quem ocupa cada
    coluna em cada trecho (as substituições trocam o jogador da coluna)."""
    base = os.path.join(pasta, "Sample_Game_{}_".format(g))
    raiz = ET.parse(base + "metadata.xml").getroot()
    for el in raiz.iter():                       # tira o namespace, se houver
        el.tag = el.tag.split("}")[-1]
    par = {p.findtext("Name"): p.findtext("Value") for p in raiz.iter("ProviderParameter")}
    time_do = {p.get("id"): p.get("teamId") for p in raiz.iter("Player")}
    times = [t.get("id") for t in raiz.iter("Team")]
    jogador_do_canal = {c.get("id"): c.get("playerId") for c in raiz.iter("PlayerChannel")}
    trechos_fmt = []
    for spec in raiz.iter("DataFormatSpecification"):
        canais = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        donos = [jogador_do_canal[c] for c in canais[0::2]]
        trechos_fmt.append((int(spec.get("startFrame")), int(spec.get("endFrame")), donos))

    with open(base + "tracking.txt") as arq:
        linhas = arq.read().split("\n")
    linhas = [ln for ln in linhas if ln]
    n = int(linhas[-1].split(":", 1)[0])
    k = len(trechos_fmt[0][2])
    jog = np.full((n, k, 2), np.nan, dtype=np.float32)
    bola = np.full((n, 2), np.nan, dtype=np.float32)
    for ln in linhas:
        f, meio, b = ln.split(":")
        i = int(f) - 1
        jog[i] = np.array(re.split("[;,]", meio), dtype=np.float32).reshape(-1, 2)[:k]
        bola[i] = np.array(b.split(","), dtype=np.float32)
    lado = np.zeros((n, k), dtype=np.int8)
    for a, b, donos in trechos_fmt:
        lado[a - 1:b] = [MANDANTE if time_do[d] == times[0] else VISITANTE for d in donos]
    tempo = np.zeros(n, dtype=np.int64)
    tempo[int(par["first_half_start"]) - 1:int(par["first_half_end"])] = 1
    tempo[int(par["second_half_start"]) - 1:int(par["second_half_end"])] = 2
    with open(base + "events.json") as arq:
        dados = json.load(arq)["data"]
    eventos = [(MANDANTE if e["team"]["id"] == times[0] else VISITANTE, e["type"]["name"],
                int(e["start"]["frame"]) - 1, int(e["end"]["frame"]) - 1, int(e["period"])) for e in dados]
    return jog, lado, bola, tempo, eventos


# ------------------------------------------------------------------------------------------------
# rótulos
def rotular(sem_bola, tempo, eventos, falta_min=FALTA_MIN, controle_min=CONTROLE_MIN):
    """(rótulos (n,) H/A/D/X, tempos [(tempo, primeiro, último)]) a partir da bola ausente e dos eventos."""
    n = len(sem_bola)
    rot = np.full(n, X, dtype=np.int8)
    tempos = []
    for t in (1, 2):
        do_tempo = sorted((a, b, time, tipo) for time, tipo, a, b, per in eventos if per == t)
        if not do_tempo:
            continue
        ini = do_tempo[0][0]
        fim = min(max(max(a, b) for a, b, _, _ in do_tempo), int(np.flatnonzero(tempo == t)[-1]))
        tempos.append((t, ini, fim))
        morta = np.zeros(n, dtype=bool)
        for a, b in trechos(sem_bola[ini:fim + 1]):
            if b - a >= falta_min:
                morta[ini + a:ini + b] = True

        dono = np.full(n, -1, dtype=np.int8)           # time do último evento com a bola
        controlada = np.zeros(n, dtype=bool)
        for a, b, time, tipo in do_tempo:
            if tipo in COM_BOLA:
                dono[a] = time
                if tipo in CONTROLADA:
                    controlada[a:max(b, a) + 1] = True
        ult = -1
        for i in range(ini, fim + 1):                  # preenche para frente; a parada zera
            if morta[i]:
                ult = -1
            elif dono[i] >= 0:
                ult = dono[i]
            dono[i] = ult
        prox = -1
        for i in range(fim, ini - 1, -1):              # entre a bola voltar e o 1º evento: o próximo time
            if morta[i]:
                prox = -1
            elif dono[i] >= 0:
                prox = dono[i]
            else:
                dono[i] = prox

        # toque sem controle: trecho curto de um time, sem ação controlada, fica com o time de antes
        for a, b in trechos(~morta[ini:fim + 1]):
            _sem_toques(dono, controlada, ini + a, ini + b, controle_min)
        seg = slice(ini, fim + 1)
        rot[seg] = np.where(morta[seg], D, np.where(dono[seg] == MANDANTE, H, np.where(dono[seg] == VISITANTE, A, D)))
    return rot, tempos


def _sem_toques(dono, controlada, a, b, controle_min):
    """Num trecho de bola viva [a, b): passa para o time de antes cada trecho curto sem ação controlada
    (menos o primeiro depois da parada). Repete até não haver mais nenhum."""
    while True:
        seg = dono[a:b]
        lim = np.r_[0, np.flatnonzero(np.diff(seg) != 0) + 1, len(seg)]
        for j in range(1, len(lim) - 1):
            s, e = a + lim[j], a + lim[j + 1]
            if e - s < controle_min and not controlada[s:e].any() and dono[s - 1] >= 0:
                dono[s:e] = dono[s - 1]
                break
        else:
            return


def carregar(id_jogo, pasta=None, usar_cache=True):
    pasta_jogo = pasta_do_jogo(id_jogo, pasta)
    g = id_jogo[-1]
    tam = sum(os.path.getsize(os.path.join(pasta_jogo, f)) for f in sorted(os.listdir(pasta_jogo)))
    tag = hashlib.sha1("{}|{}|{}|{}".format(id_jogo, tam, FALTA_MIN, CONTROLE_MIN).encode()).hexdigest()[:10]
    cache = os.path.join(config.CACHE, "metrica_{}_{}.npz".format(id_jogo, tag))
    if usar_cache and os.path.isfile(cache):
        z = config.ler_npz(cache)
        return Partida(jogadores=z["jogadores"], lado=z["lado"], bola=z["bola"], campo=CAMPO,
                       tempos=[tuple(int(v) for v in t) for t in z["tempos"]], rotulos=z["rotulos"], nome=id_jogo)
    jog, lado, bola, tempo, eventos = _ler_epts(pasta_jogo, g) if g == "3" else _ler_csv(pasta_jogo, g)
    n = len(bola)
    if lado.ndim == 1:
        lado = np.broadcast_to(lado, (n, len(lado))).copy()
    rot, tempos = rotular(np.isnan(bola[:, 0]), tempo, eventos)
    escala = np.array(CAMPO, dtype=np.float32)
    jog, bola = jog * escala, bola * escala
    if usar_cache:
        config.gravar_npz(cache, jogadores=jog, lado=lado, bola=bola, rotulos=rot, tempos=np.array(tempos))
    return Partida(jogadores=jog, lado=lado, bola=bola, campo=CAMPO, tempos=tempos, rotulos=rot, nome=id_jogo)


def resumo(rot):
    """Estatísticas dos rótulos, para comparar com o GT público."""
    jog = rot != X
    posses = dur = 0
    for c in (H, A):
        for a, b in trechos(rot == c):
            posses += 1
            dur += b - a
    minutos = jog.sum() / config.FPS / 60.0
    return {"minutos": minutos, "bola_morta": float(np.mean(rot[jog] == D)),
            "posses_por_min": posses / minutos, "duracao_media_s": dur / max(posses, 1) / config.FPS,
            "mandante": float(np.mean(rot[(rot == H) | (rot == A)] == H))}


def main(argv=None):
    import sys
    linhas = []
    if os.path.isfile(config.GT_PUBLICO):
        gt = pd.read_csv(config.GT_PUBLICO)["label"].map({"H": H, "A": A, "D": D, "X": X}).to_numpy()
        linhas.append(("GT público (5 min)", resumo(gt)))
    achados = jogos()
    if not achados:
        sys.exit("nenhum jogo da Metrica em {} (python -m posse_ml.baixar_metrica)".format(config.METRICA))
    for j in achados:
        linhas.append((j, resumo(carregar(j).rotulos)))
    print("{:20s} {:>8s} {:>11s} {:>11s} {:>12s} {:>9s}".format(
        "", "minutos", "bola morta", "posses/min", "duração (s)", "mandante"))
    for nome, r in linhas:
        print("{:20s} {:8.1f} {:11.1%} {:11.2f} {:12.1f} {:9.1%}".format(
            nome, r["minutos"], r["bola_morta"], r["posses_por_min"], r["duracao_media_s"], r["mandante"]))


if __name__ == "__main__":
    main()
