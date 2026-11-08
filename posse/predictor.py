"""Preditor de posse: H, A ou D em cada frame do vídeo tático.

    python posse/predictor.py --dead <bola_morta.csv> --ball <bola.csv> --out <submissão.csv>

    # teste nos 5 minutos públicos, com as fronteiras de bola morta do próprio GT:
    python posse/predictor.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha \
        --ball <bola.csv> --out posse/out/teste_publico.csv

A bola morta vem do detector da equipe (--dead). Dentro de cada trecho de bola viva, a posse é
decidida por um Viterbi com dois estados, H e A:

    custo(sequência) = soma dos custos por frame + SWITCH_COST por troca de time

e começa no dono do reinício (restart_owner.py). O custo de estar no time X num frame é a evidência,
naquele frame, a favor do outro time. Três fontes de evidência:

  1. TOQUE. Na bola suavizada, um toque é um pico local da mudança do vetor velocidade entre os
     frames antes e depois (Vidal-Codina et al., 2022, como na DataBallPy: um jogador só ganha a
     bola se ela muda de direção ou de velocidade perto dele; o vetor junta os dois critérios). O
     toque vai para o time do jogador mais perto, se ele estiver a até TOUCH_RADIUS e com folga de
     TOUCH_MARGIN sobre o adversário; senão é bola disputada e não conta, porque na disputa a posse
     fica com quem já tinha (CHALLENGE.md). Pelo GT público, a posse troca praticamente a cada toque
     do adversário, então um toque claro paga uma troca (TOUCH_WEIGHT > SWITCH_COST). Mas não paga
     duas sozinho: como ~1 em 10 posições da bola está errada, uma posse curta no meio de outra
     exige o toque mais alguma confirmação (o adversário perto da bola por alguns frames).
  2. PROXIMIDADE. Para quando o toque não é visto (a bola some justamente nos lances disputados).
     A cada frame com bola, o time do jogador mais perto ganha até PROX_WEIGHT, decaindo com a
     distância e com a folga sobre o adversário. Bola em voo, longe de todos, não vota. Cerca de 1 s
     de domínio claro soma o bastante para trocar sem ter visto o toque.
  3. ÚLTIMO TOQUE. Desempate fraco: enquanto a bola viaja sem evidência, favorece o time que tocou
     por último. Faz a troca cair na recepção do adversário, e não no passe.

A bola passa antes por um filtro: posições a mais de SPIKE_DIST da mediana dos vizinhos (pulos para
outro objeto, o erro típico do tracker) são descartadas, porque cada uma criaria um toque falso. Não
há interpolação: onde não há bola, não há evidência e o Viterbi mantém a posse, que é o mesmo efeito
do X no spellF1 e é melhor no macroF1 (decisão do tracking_bola/README.md, passo 15 e seção 6).

Saídas:
  - a submissão (D na bola morta, H/A no vivo, X fora das metades);
  - posse/out/toques.csv, uma camada para o viewer ("Adicionar tracking"): a bola em cada toque
    detectado, colorida pelo time;
  - no terminal, com o GT público, spellF1 e macroF1 do avaliador oficial e cada troca do GT ao lado
    da troca prevista mais próxima.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import restart_owner as ro  # noqa: E402

# Params -- unidades: metros, segundos a 25 fps, custos adimensionais
SWITCH_COST = 1.0      # custo de cada troca H <-> A
START_COST = 5.0       # custo de começar o trecho no time que não é o dono do reinício
START_COST_WEAK = 1.0  # idem, quando o dono do reinício foi decidido com pouca evidência

SPIKE_SPAN = 5         # frames de cada lado na mediana usada para achar posições que pulam
SPIKE_DIST = 3.0       # m: posição mais longe que isso da mediana dos vizinhos é descartada
SMOOTH = 5             # janela da média móvel da posição da bola (frames)
VEL_SPAN = 4           # frames antes e depois usados para a velocidade de entrada e de saída
VEL_MIN = 3            # velocidades válidas exigidas de cada lado para medir um toque
DV_MIN = 4.0           # m/s: piso da mudança do vetor velocidade para contar um toque
DV_NOISE_FACTOR = 3.0  # ...e pelo menos 3x a mediana dessa mudança no jogo (o ruído típico)
TOUCH_RADIUS = 2.0     # m: o tocador está a até isso da bola (DataBallPy usa 1,5 m em dados limpos)
TOUCH_MARGIN = 0.75    # m de folga sobre o adversário mais próximo; senão é disputa
AFTER_FRAMES = 12      # frames depois do toque (~0,5 s) para ver quem sai com a bola
AFTER_MIN_FRAMES = 6   # frames com distâncias medidas exigidos nessa janela
AFTER_RADIUS = 2.0     # m: distância média até a bola de quem "sai com ela"
TOUCH_WEIGHT = 1.5     # sem o portão: um toque claro paga uma troca (1), mas não duas sozinho
TOUCH_WEIGHT_GATED = 0.5  # com o portão: o toque libera a troca, mas sozinho não a paga

# Portão: o toque do novo time é condição NECESSÁRIA para a troca (não suficiente)
GATE_NOISE_FACTOR = 2.0   # eventos do portão: picos de dv acima de 2x o ruído típico...
GATE_DV_MIN = 3.0         # ...e de pelo menos 3 m/s
GATE_RADIUS = 3.0         # m: algum jogador do novo time a até isso da bola, mesmo sem ser o mais perto
GATE_WINDOW = 25          # frames (1 s) de cada lado do evento em que a troca fica liberada
FLIGHT_SPEED = 12.0       # m/s: bola rápida antes E depois...
FLIGHT_ANGLE = 30.0       # ...e com menos que isso de mudança de direção = bola em voo, não toque
REAPPEAR_GAP = 10         # frames: lacuna da bola a partir da qual o reaparecimento pode liberar troca
REAPPEAR_LOOK = 5         # frames depois do reaparecimento olhados para ver quem domina a bola

PROX_WEIGHT = 0.08     # evidência máxima por frame da proximidade
PROX_RADIUS = 2.5      # m: escala do decaimento com a distância
PROX_MARGIN = 1.0      # m: escala da folga sobre o adversário
CONTROL_SPEED = 7.0    # m/s: bola mais rápida que isso está em trânsito; a proximidade não conta

LAST_TOUCH_WEIGHT = 0.01  # desempate por frame a favor de quem tocou por último

TAG_WEIGHT = 0.0       # evidência por frame das fases do Smart Tagging (0 = desligado)

# Toques aéreos (--aerial): a câmera vem da homografia; a bola no ar está em algum ponto do raio
# entre a câmera e a projeção dela no chão.
IMAGE_SIZE = (1920, 1080)  # vídeo tático (DATA.md); o centro óptico é suposto no meio da imagem
REACH_HEIGHT = 2.5     # m: altura máxima em que um jogador toca a bola (cabeçada com salto)
AIR_RADIUS = 1.0       # m: distância máxima do corpo do jogador ao raio da bola, no toque estrito
AIR_GATE_RADIUS = 1.5  # m: idem, para liberar a troca no portão
CAM_HEIGHT_RANGE = (3.0, 120.0)  # m: alturas de câmera plausíveis; fora disso o frame é descartado

KICKOFFS = {1: 11185, 2: 113742}  # kickoff de cada tempo (DATA.md / FAQ.md): define o gol de cada time
GK_MIN_FRAMES = 1000   # frames presentes para um slot poder ser escolhido como goleiro
GK_MIN_FRACTION = 0.5  # o goleiro é o mais recuado do time em pelo menos metade dos frames

OUT_DIR = os.path.join(HERE, "out")


# ------------------------------------------------------------------------------------------------
# Dados
# ------------------------------------------------------------------------------------------------

_ATD_CACHE = {}


def read_atd_players(raw_path, n_slots, n_frames):
    """Jogadores em todos os frames, (n_frames, n_slots, 2) float32, coordenadas 0-1.

    Guardado em memória por caminho: várias rodadas no mesmo processo (comparar_bolas.py) leem o
    ATD uma vez só.
    """
    key = (raw_path, n_slots, n_frames)
    if key in _ATD_CACHE:
        return _ATD_CACHE[key]
    _ATD_CACHE[key] = _read_atd_players(raw_path, n_slots, n_frames)
    return _ATD_CACHE[key]


def _read_atd_players(raw_path, n_slots, n_frames):
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            ball[frame] = ro._xy(parts[2])
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                players[frame, slot] = ro._xy(group)
    return players, ball


class PlayersView:
    """Adapta o array de jogadores à interface que restart_owner.decide espera (frame -> array)."""

    def __init__(self, players):
        self.players = players

    def __contains__(self, frame):
        return 0 <= frame < len(self.players) and np.isfinite(self.players[frame, :, 0]).any()

    def __getitem__(self, frame):
        return self.players[frame].astype(float)


def pixels_to_field(ball_px):
    """Pixels do vídeo -> coordenadas normalizadas do campo (0-1), pela inversa da homografia.

    A homografia do tactical mapeia campo (0-1) em pixels e é válida em todos os frames (DATA.md);
    onde não for, o frame fica sem bola. É a mesma conversão do tracking_bola/export_ball.py.
    """
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    from load_homography import load as load_homography
    matrices, _, valid = load_homography()
    out = np.full_like(ball_px, np.nan)
    ok = np.isfinite(ball_px[:, 0]) & valid[:len(ball_px)]
    points = np.column_stack([ball_px[ok], np.ones(int(ok.sum()))])
    mapped = np.einsum("nij,nj->ni", np.linalg.inv(matrices[:len(ball_px)][ok]), points)
    out[ok] = mapped[:, :2] / mapped[:, 2:3]
    return out


def despike(ball_m, span, max_dist):
    """Descarta posições a mais de max_dist da mediana dos vizinhos (±span frames).

    O tracker da bola erra cerca de 1 em 10 posições que dá, em geral pulando para outro objeto por
    poucos frames (tracking_bola/README.md). A mediana, coordenada a coordenada, acompanha a bola em
    linha reta e também na quebra de direção de um toque real; um pulo de até `span` frames não a
    arrasta e fica longe dela. Precisa de pelo menos 3 vizinhos com bola; senão a posição fica.
    """
    out = ball_m.copy()
    known = np.isfinite(ball_m[:, 0])
    for t in np.flatnonzero(known):
        window = ball_m[max(0, t - span):t + span + 1]
        window = window[np.isfinite(window[:, 0])]
        if len(window) < 3:
            continue
        if np.hypot(*(ball_m[t] - np.median(window, axis=0))) > max_dist:
            out[t] = np.nan
    return out


def smooth(ball, window):
    """Média móvel centrada, ignorando NaN; frame sem bola continua sem bola."""
    half = window // 2
    known = np.isfinite(ball[:, 0])
    values = np.where(known[:, None], ball, 0.0)
    total = np.cumsum(np.vstack([np.zeros((1, 2)), values]), axis=0)
    count = np.cumsum(np.concatenate([[0], known.astype(int)]))
    idx = np.arange(len(ball))
    lo, hi = np.maximum(0, idx - half), np.minimum(len(ball), idx + half + 1)
    mean = (total[hi] - total[lo]) / np.maximum(count[hi] - count[lo], 1)[:, None]
    mean[~known] = np.nan
    return mean


def nanmean_window(values, start, stop):
    """Média de values[start:stop] (vetores), NaN se houver menos de VEL_MIN válidos."""
    piece = values[max(0, start):max(0, stop)]
    piece = piece[np.isfinite(piece[:, 0])]
    return piece.mean(axis=0) if len(piece) >= VEL_MIN else np.array([np.nan, np.nan])


# ------------------------------------------------------------------------------------------------
# Evidência
# ------------------------------------------------------------------------------------------------

def team_distances(players_m, ball_m, teams):
    """(d_H, d_A): distância da bola ao jogador mais perto de cada time, por frame. inf se faltar.

    teams é o time de cada slot, fixo (n_slots,) ou por frame (n_frames, n_slots).
    """
    diff = players_m - ball_m[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    teams = np.broadcast_to(teams, dist.shape)
    return tuple(np.where(teams == side, dist, np.inf).min(axis=1) for side in (ro.HOME, ro.AWAY))


def camera_from_homography(matrices, pitch, image_size=IMAGE_SIZE):
    """(centro da câmera em metros (n, 3), foco em px (n,)) a partir das homografias campo->imagem.

    A homografia do plano do campo (Z = 0) é H = K [r1 r2 t], a menos de escala. Com o centro óptico
    no meio da imagem e pixels quadrados, K só tem o foco f, que sai das duas restrições de
    ortonormalidade de r1 e r2 (calibração de Zhang com um plano). Depois, r3 = r1 x r2 e o centro
    da câmera é C = -R^T t, com Z para cima (o sinal da homografia é escolhido para C_z > 0).
    Frames sem solução (foco imaginário, altura implausível) ficam NaN.
    """
    length, width = pitch
    n = len(matrices)
    shift = np.array([[1, 0, -image_size[0] / 2.0], [0, 1, -image_size[1] / 2.0], [0, 0, 1]])
    metric = shift @ matrices @ np.diag([1.0 / length, 1.0 / width, 1.0])  # metros -> px centrados
    h1, h2, h3 = metric[:, :, 0], metric[:, :, 1], metric[:, :, 2]
    a1 = h1[:, 0] * h2[:, 0] + h1[:, 1] * h2[:, 1]
    b1 = h1[:, 2] * h2[:, 2]
    a2 = h1[:, 0] ** 2 + h1[:, 1] ** 2 - h2[:, 0] ** 2 - h2[:, 1] ** 2
    b2 = h1[:, 2] ** 2 - h2[:, 2] ** 2
    w = -(a1 * b1 + a2 * b2) / (a1 ** 2 + a2 ** 2)  # w = 1 / f²
    centers = np.full((n, 3), np.nan)
    focal = np.full(n, np.nan)
    ok = np.isfinite(w) & (w > 0)
    for i in np.flatnonzero(ok):
        f = 1.0 / np.sqrt(w[i])
        k_inv = np.diag([1.0 / f, 1.0 / f, 1.0])
        g1, g2, g3 = k_inv @ h1[i], k_inv @ h2[i], k_inv @ h3[i]
        lam = 2.0 / (np.linalg.norm(g1) + np.linalg.norm(g2))
        for sign in (1.0, -1.0):
            r1, r2, t = sign * lam * g1, sign * lam * g2, sign * lam * g3
            u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
            rot = u @ vt
            c = -rot.T @ t
            if c[2] > 0:
                break
        if CAM_HEIGHT_RANGE[0] <= c[2] <= CAM_HEIGHT_RANGE[1]:
            centers[i], focal[i] = c, f
    return centers, focal


def estimate_camera(blocks, pitch, n_frames):
    """Centro da câmera por frame (n_frames, 3), com a mediana de cada tempo, e um resumo.

    A câmera tática fica num lugar fixo do estádio: a mediana por tempo é robusta a frames ruins.
    Devolve (None, motivo) se a homografia não puder ser lida.
    """
    try:
        sys.path.insert(0, os.path.join(ROOT, "starter"))
        from load_homography import load as load_homography
        matrices, _, valid = load_homography()
    except Exception as error:  # sem homografia, sem toques aéreos
        return None, "homografia indisponível ({})".format(error)
    centers = np.full((n_frames, 3), np.nan)
    lines = []
    for half, first, last in blocks:
        idx = np.arange(first, min(last + 1, len(matrices)))
        idx = idx[valid[idx]]
        c, f = camera_from_homography(matrices[idx], pitch)
        good = np.isfinite(c[:, 2])
        if good.sum() < 100:
            return None, "câmera não recuperada no {}º tempo".format(half)
        med = np.median(c[good], axis=0)
        q1, q3 = np.percentile(c[good], [25, 75], axis=0)
        centers[first:last + 1] = med
        lines.append("{}º tempo: x {:.1f} m, y {:.1f} m, altura {:.1f} m (intervalo interquartil "
                     "{:.1f} / {:.1f} / {:.1f} m), foco {:.0f} px, {:.0%} dos frames resolvidos".format(
                         half, med[0], med[1], med[2], q3[0] - q1[0], q3[1] - q1[1], q3[2] - q1[2],
                         np.median(f[good]), good.mean()))
    return centers, "; ".join(lines)


def air_distances(players_m, ball_m, teams, centers):
    """(dH, dA): menor distância do corpo de algum jogador de cada time ao raio da bola, por frame.

    A bola no ar está no raio entre a câmera C e a projeção dela no chão G. Até REACH_HEIGHT de
    altura, esse raio cobre no chão o segmento de G até G + (REACH_HEIGHT / C_z) (C_h - G), e cada
    ponto dele está a uma altura que um jogador alcança. A distância de um jogador (uma vertical
    dos pés à cabeça) a esse raio é a distância dele, no chão, a esse segmento. Para a bola no chão,
    o segmento começa no ponto da bola, então essa distância nunca é maior que a do chão.
    """
    seg = (REACH_HEIGHT / centers[:, 2:3]) * (centers[:, :2] - ball_m)  # (n, 2)
    rel = players_m - ball_m[:, None, :]                                 # (n, S, 2)
    length2 = np.maximum((seg ** 2).sum(axis=1), 1e-9)[:, None]
    s = np.clip((rel * seg[:, None, :]).sum(axis=2) / length2, 0.0, 1.0)
    diff = rel - s[..., None] * seg[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isfinite(dist), dist, np.inf)
    teams = np.broadcast_to(teams, dist.shape)
    return tuple(np.where(teams == side, dist, np.inf).min(axis=1) for side in (ro.HOME, ro.AWAY))


def home_defends_left(players, team, half, first, last):
    """O mandante defende o gol da esquerda (x = 0) nesta metade? Decidido no kickoff.

    No kickoff todos estão no próprio campo, então o time com o x médio menor defende a esquerda
    (como no diagnostico/verificar_goleiro.py; a média do tempo inteiro fica perto demais de 0,5).
    """
    kick = KICKOFFS.get(half, first)
    if not first <= kick <= last:
        kick = first
    kx = players[kick, :, 0].astype(float)
    return np.nanmean(kx[team == ro.HOME]) < np.nanmean(kx[team == ro.AWAY])


def find_goalkeepers(players, team, blocks, pitch):
    """{time: (slot, fração)}: em cada time, o slot que mais vezes é o mais recuado do próprio time.

    Mesmo critério do diagnostico/verificar_goleiro.py: fração dos frames avaliados em que o slot,
    presente, é o jogador do seu time mais perto do próprio gol. Só slots com pelo menos
    GK_MIN_FRAMES frames presentes. Neste jogo dá os slots 24 e 46 (96% e 86%), e nenhum outro
    passa de 7%.
    """
    length = pitch[0]
    deepest = np.zeros(players.shape[1])
    present = np.zeros(players.shape[1])
    for half, first, last in blocks:
        x = players[first:last + 1, :, 0].astype(float) * length
        home_left = home_defends_left(players, team, half, first, last)
        present += np.isfinite(x).sum(axis=0)
        for side in (ro.HOME, ro.AWAY):
            left = home_left if side == ro.HOME else not home_left
            depth = np.where(team == side, x if left else length - x, np.nan)
            depth = np.where(np.isfinite(depth), depth, np.inf)
            rows = np.flatnonzero(np.isfinite(depth).any(axis=1))
            np.add.at(deepest, np.argmin(depth[rows], axis=1), 1)
    out = {}
    for side in (ro.HOME, ro.AWAY):
        frac = np.where((team == side) & (present >= GK_MIN_FRAMES),
                        deepest / np.maximum(present, 1), -1.0)
        slot = int(np.argmax(frac))
        out[side] = (slot, float(frac[slot]))
    return out


def goalkeeper_teams(players, team, blocks, pitch):
    """Time de cada slot por frame, com o time dos goleiros dado pela posição.

    Os slots dos goleiros trocam de pessoa: em alguns trechos o slot do goleiro do Athletic passa
    a seguir o do Barcelona, com o time errado (diagnostico/README.md, passos 2 e 4). Regra: os
    dois slots de goleiro (find_goalkeepers) recebem, em cada frame, o time que defende o gol do
    lado do campo em que estão. Todos os outros slots mantêm o time do ATD: um jogador de linha
    que fica mais recuado que o goleiro (lance na pequena área, goleiro adiantado) não é afetado.

    Devolve (times por frame, {slot: frames em que o time mudou}, goleiros).
    """
    length = pitch[0]
    n_frames, n_slots = players.shape[:2]
    teams = np.broadcast_to(team, (n_frames, n_slots)).copy()
    keepers = {side: (slot, frac) for side, (slot, frac)
               in find_goalkeepers(players, team, blocks, pitch).items()
               if frac >= GK_MIN_FRACTION}  # sem goleiro claro, nada é corrigido nesse time
    changed = {}
    for half, first, last in blocks:
        home_left = home_defends_left(players, team, half, first, last)
        for slot, _ in keepers.values():
            x = players[first:last + 1, slot, 0].astype(float) * length
            ok = np.isfinite(x)
            on_left = x < length / 2
            owner = np.where(on_left == home_left, ro.HOME, ro.AWAY)
            frames = first + np.flatnonzero(ok)
            new = owner[ok]
            changed[slot] = changed.get(slot, 0) + int(np.sum(teams[frames, slot] != new))
            teams[frames, slot] = new
    return teams, changed, keepers


def velocity_change(ball_m, segments, fps):
    """(dv, v_antes, v_depois) por frame: dv = |v_depois - v_antes| (m/s); NaN sem dados."""
    velocity = np.full_like(ball_m, np.nan)
    velocity[1:-1] = (ball_m[2:] - ball_m[:-2]) * fps / 2.0
    v_in = np.full_like(ball_m, np.nan)
    v_out = np.full_like(ball_m, np.nan)
    for _, start, end in segments:
        for t in range(start + 1, end - 1):
            v_in[t] = nanmean_window(velocity, t - VEL_SPAN, t)
            v_out[t] = nanmean_window(velocity, t + 1, t + 1 + VEL_SPAN)
    return np.hypot(*(v_out - v_in).T), v_in, v_out


def local_peaks(dv, segments, threshold):
    """Frames de pico local de dv (±3 frames) acima do limiar, dentro dos trechos vivos."""
    peaks = []
    for _, start, end in segments:
        piece = dv[start:end]
        for i in np.flatnonzero(piece >= threshold):
            if piece[i] >= np.nanmax(piece[max(0, i - 3):i + 4]):
                peaks.append(start + int(i))
    return peaks


def in_flight(v_in, v_out):
    """Bola rápida antes e depois, quase sem mudar de direção: em voo, não foi tocada.

    A homografia vale para o chão: a projeção de uma bola alta não anda em linha reta com
    velocidade constante, e parece acelerar ou curvar sem ninguém tocá-la (lance 12757).
    """
    s1, s2 = np.hypot(*v_in), np.hypot(*v_out)
    if not (s1 >= FLIGHT_SPEED and s2 >= FLIGHT_SPEED):
        return False
    cos = np.clip(np.dot(v_in, v_out) / (s1 * s2), -1.0, 1.0)
    return np.degrees(np.arccos(cos)) < FLIGHT_ANGLE


def touch_threshold(dv):
    """Limiar de toque: o piso DV_MIN, ou DV_NOISE_FACTOR x a mediana de dv se o ruído for maior.

    A maioria dos frames vivos não tem toque, então a mediana de dv mede o ruído da própria bola.
    Assim o limiar se adapta à bola usada (ATD ou fusão) sem olhar rótulo nenhum.
    """
    noise = float(np.nanmedian(dv)) if np.isfinite(dv).any() else 0.0
    return max(DV_MIN, DV_NOISE_FACTOR * noise), noise


def attribute_after(t, d_home, d_away, end):
    """Time que sai COM a bola depois do toque: o mais perto, em média, de t até t + AFTER_FRAMES.

    Numa recepção sob marcação, recebedor e marcador podem estar à mesma distância no instante do
    toque, e um metro de erro no tracking inverte quem parece mais perto (lance 11349). Logo
    depois, a bola sai com quem a recebeu e o marcador fica para trás. Devolve o time, ou None se
    ninguém fica com a bola (passe ou chute: a bola se afasta de todos) ou se não dá para separar.
    """
    hi = min(end, t + AFTER_FRAMES + 1)
    dh, da = d_home[t:hi], d_away[t:hi]
    ok = np.isfinite(dh) & np.isfinite(da)
    if ok.sum() < AFTER_MIN_FRAMES:
        return None
    mh, ma = float(np.mean(dh[ok])), float(np.mean(da[ok]))
    near = ro.HOME if mh < ma else ro.AWAY
    if min(mh, ma) <= AFTER_RADIUS and abs(mh - ma) >= TOUCH_MARGIN:
        return near
    return None


def near_jump(t, jumpy):
    """Há uma posição que pula a até SPIKE_SPAN frames de t?"""
    return jumpy is not None and bool(jumpy[max(0, t - SPIKE_SPAN):t + SPIKE_SPAN + 1].any())


def aerial_owner(t, air_home, air_away, radius):
    """Time cujo jogador está a até radius do raio da bola (com folga), em t-2..t+2; ou None."""
    lo, hi = max(0, t - 2), t + 3
    ah, aa = air_home[lo:hi].min(), air_away[lo:hi].min()
    near = ro.HOME if ah < aa else ro.AWAY
    if min(ah, aa) <= radius and max(ah, aa) - min(ah, aa) >= TOUCH_MARGIN:
        return near
    return None


def detect_touches(dv, v_in, v_out, d_home, d_away, segments, flight_filter, use_after=False,
                   air=None, jumpy=None):
    """Toques ESTRITOS (evidência): [(frame, time, dv)], com o limiar, o ruído e os frames aéreos.

    Atribuição: o mais perto no instante, a até TOUCH_RADIUS e com folga TOUCH_MARGIN sobre o
    adversário. Com use_after, antes disso tenta quem sai com a bola nos frames seguintes.
    Com air = (dH_ar, dA_ar): se NINGUÉM está a até TOUCH_RADIUS da bola no chão, a quebra de
    trajetória não pode ter sido no chão; a bola estava no ar, e o toque vai para quem tem o corpo a
    até AIR_RADIUS do raio da bola (cabeça, peito). Se alguém está perto no chão, vale o chão: uma
    câmera só não distingue uma bola alta de uma bola no chão atrás de outro jogador.
    jumpy (bool por frame): perto de uma posição que pula (filtro de pulos), a quebra pode ser um
    pulo para outro objeto, não um toque; aí não há toque aéreo.
    """
    threshold, noise = touch_threshold(dv)
    touches, aerial = [], set()
    segment_end = np.zeros(len(dv), dtype=int)
    for _, start, end in segments:
        segment_end[start:end] = end
    for t in local_peaks(dv, segments, threshold):
        if flight_filter and in_flight(v_in[t], v_out[t]):
            continue
        after = attribute_after(t, d_home, d_away, segment_end[t]) if use_after else None
        if after is not None:
            touches.append((t, after, float(dv[t])))
            continue
        lo, hi = max(0, t - 2), t + 3
        dh, da = d_home[lo:hi].min(), d_away[lo:hi].min()
        near = ro.HOME if dh < da else ro.AWAY
        d_near, d_far = min(dh, da), max(dh, da)
        if d_near <= TOUCH_RADIUS and d_far - d_near >= TOUCH_MARGIN:
            touches.append((t, near, float(dv[t])))
        elif air is not None and d_near > TOUCH_RADIUS and not near_jump(t, jumpy):
            owner = aerial_owner(t, air[0], air[1], AIR_RADIUS)
            if owner is not None:
                touches.append((t, owner, float(dv[t])))
                aerial.add(t)
    return touches, threshold, noise, aerial


def gate_mask(dv, v_in, v_out, d_home, d_away, ball_m, segments, noise, air=None, jumpy=None):
    """allow[time, frame]: onde a troca PARA esse time é permitida (toque = condição necessária).

    1. Eventos permissivos: picos de dv acima de GATE_NOISE_FACTOR x o ruído, sem os de voo. Cada
       evento libera, por ±GATE_WINDOW frames, a troca para todo time que tenha ALGUM jogador a até
       GATE_RADIUS da bola, mesmo sem ser o mais perto (em 11351 e 15766 o recebedor do Barça
       estava a 2-3 m e um marcador do Athletic mais perto).
    2. Reaparecimento: quando a bola some por REAPPEAR_GAP frames ou mais e reaparece com um time
       claramente nela (critério do toque estrito), a troca para esse time fica liberada em toda a
       lacuna e até GATE_WINDOW depois: o toque aconteceu no escuro.
    """
    n = len(dv)
    allow = np.zeros((2, n), dtype=bool)
    threshold = max(GATE_DV_MIN, GATE_NOISE_FACTOR * noise)
    stats = {"eventos": 0, "voo": 0, "reaparecimentos": 0, "limiar": threshold, "aereos": 0,
             "frames": ([], [])}  # frames dos eventos que liberam cada time (para a ancoragem)
    for t in local_peaks(dv, segments, threshold):
        if in_flight(v_in[t], v_out[t]):
            stats["voo"] += 1
            continue
        stats["eventos"] += 1
        lo, hi = max(0, t - 2), t + 3
        nobody_on_ground = (min(d_home[lo:hi].min(), d_away[lo:hi].min()) > GATE_RADIUS
                            and not near_jump(t, jumpy))
        for side, dist in ((ro.HOME, d_home), (ro.AWAY, d_away)):
            ok = dist[lo:hi].min() <= GATE_RADIUS
            if not ok and nobody_on_ground and air is not None:  # bola no ar: corpo perto do raio
                ok = air[side][lo:hi].min() <= AIR_GATE_RADIUS
                stats["aereos"] += int(ok)
            if ok:
                allow[side, max(0, t - GATE_WINDOW):t + GATE_WINDOW + 1] = True
                stats["frames"][side].append(t)

    known = np.isfinite(ball_m[:, 0])
    for _, start, end in segments:
        missing = ~known[start:end]
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b - a < REAPPEAR_GAP or start + b >= end:
                continue
            r = start + int(b)  # primeiro frame com bola depois da lacuna
            dh = d_home[r:r + REAPPEAR_LOOK].min()
            da = d_away[r:r + REAPPEAR_LOOK].min()
            near = ro.HOME if dh < da else ro.AWAY
            if min(dh, da) <= TOUCH_RADIUS and max(dh, da) - min(dh, da) >= TOUCH_MARGIN:
                allow[near, start + int(a):r + GATE_WINDOW + 1] = True
                stats["reaparecimentos"] += 1
    return allow, stats


def ball_speed(ball_m, fps):
    """Velocidade da bola por frame (m/s), por diferença central; NaN sem os dois vizinhos."""
    speed = np.full(len(ball_m), np.nan)
    speed[1:-1] = np.hypot(*((ball_m[2:] - ball_m[:-2]) * fps / 2.0).T)
    return speed


def proximity_evidence(d_home, d_away, speed=None):
    """(ev_H, ev_A) por frame: o time mais perto da bola, pesado pela distância e pela folga.

    Com speed, só contam os frames com a bola SOB CONTROLE (até CONTROL_SPEED). Bola em trânsito
    (passe, chutão) não é de ninguém: sem isso, o jogador do outro time que espera a bola, ou se
    aproxima para interceptá-la, somava evidência antes do toque, e as trocas saíam ~1 s cedo.
    """
    ev = np.zeros((2, len(d_home)))
    near_home = d_home < d_away
    d_near = np.where(near_home, d_home, d_away)
    d_far = np.where(near_home, d_away, d_home)
    ok = np.isfinite(d_near)
    margin = np.full(len(d_home), 3 * PROX_MARGIN)  # adversário ausente: folga grande
    both = ok & np.isfinite(d_far)
    margin[both] = d_far[both] - d_near[both]
    weight = np.zeros(len(d_home))
    weight[ok] = (PROX_WEIGHT * np.exp(-(d_near[ok] / PROX_RADIUS) ** 2)
                  * np.tanh(margin[ok] / PROX_MARGIN))
    if speed is not None:
        weight[~(speed <= CONTROL_SPEED)] = 0.0  # em trânsito ou sem velocidade medida
    ev[ro.HOME] = np.where(near_home, weight, 0.0)
    ev[ro.AWAY] = np.where(~near_home, weight, 0.0)
    return ev


PHASE_SIGN = {  # +1: o time da tag tem a bola; -1: o time da tag está defendendo
    "build up": 1, "progression": 1, "creating chances": 1, "attacking transition": 1,
    "ball in final third": 1, "ball in the box": 1, "players in the box": 1, "long ball": 1,
    "defending in attacking third": -1, "defending in middle third": -1,
    "defending in defensive third": -1, "defensive transition": -1,
}


def tag_evidence(path, n_frames, fps):
    """(ev_H, ev_A) das fases do Smart Tagging, TAG_WEIGHT por frame coberto."""
    import xml.etree.ElementTree as ET
    ev = np.zeros((2, n_frames))
    for instance in ET.parse(path).iter("instance"):
        sign = PHASE_SIGN.get((instance.findtext("code") or "").strip().lower())
        if not sign:
            continue
        team = ""
        for label in instance.findall("label"):
            if label.findtext("group") == "Team":
                team = (label.findtext("text") or "").strip().lower()
        side = ro.HOME if "barcel" in team else ro.AWAY if "atl" in team else None
        if side is None:
            continue
        owner = side if sign > 0 else 1 - side
        a = max(0, int(round(float(instance.findtext("start")) * fps)))
        b = min(n_frames, int(round(float(instance.findtext("end")) * fps)))
        ev[owner, a:b] += TAG_WEIGHT
    return ev


# ------------------------------------------------------------------------------------------------
# Decodificação
# ------------------------------------------------------------------------------------------------

def viterbi_segment(cost, start_state, start_cost, allow=None):
    """Sequência H/A de custo mínimo num trecho. cost: (2, T), custo de estar em cada estado.

    allow (2, T), opcional: trocar PARA o estado s no frame t só é possível se allow[s, t].
    """
    length = cost.shape[1]
    total = cost[:, 0].copy()
    if start_state is not None:
        total[1 - start_state] += start_cost
    back = np.zeros((2, length), dtype=np.int8)
    for t in range(1, length):
        stay, switch = total, total[::-1] + SWITCH_COST
        if allow is not None:
            switch = np.where(allow[:, t], switch, np.inf)
        back[:, t] = np.where(stay <= switch, [0, 1], [1, 0])
        total = np.minimum(stay, switch) + cost[:, t]
    path = np.empty(length, dtype=np.int8)
    path[-1] = int(np.argmin(total))
    for t in range(length - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def predict(segments, ev, last_touch_team, restart_rows, allow=None):
    """Rótulos H/A em cada trecho vivo."""
    labels = {}
    for (half, start, end), row in zip(segments, restart_rows):
        cost = np.vstack([ev[ro.AWAY, start:end], ev[ro.HOME, start:end]])  # custo de H, de A
        lt = last_touch_team[start:end]
        cost[ro.HOME] += LAST_TOUCH_WEIGHT * (lt == ro.AWAY)
        cost[ro.AWAY] += LAST_TOUCH_WEIGHT * (lt == ro.HOME)
        owner = {"H": ro.HOME, "A": ro.AWAY}.get(row["owner"])
        weak = row["source"] in ("tracking_fraco", "nenhum")
        path = viterbi_segment(cost, owner, START_COST_WEAK if weak else START_COST,
                               None if allow is None else allow[:, start:end])
        labels[(start, end)] = np.where(path == ro.HOME, "H", "A")
    return labels


def explain(first, last, dv, v_in, v_out, d_home, d_away, ev, touches, allow, labels, threshold,
            gate_threshold, segments, air=None, aerial=()):
    """Imprime, para os frames first..last, por que o preditor trocou ou não a posse.

    1. Cada pico de dv acima do limiar do portão: velocidades, ângulo, distâncias de cada time no
       instante e em média nos AFTER_FRAMES seguintes, se é voo, e o que virou (toque estrito de
       quem, ou só evento do portão).
    2. De 5 em 5 frames: rótulo previsto, evidência somada para H e para A, distâncias médias de
       cada time à bola, fração de frames com bola e se a troca estava liberada para cada time.
    """
    strict = {t: side for t, side, _ in touches}
    segment_end = np.zeros(len(dv), dtype=int)
    for _, start, end in segments:
        segment_end[start:end] = end
    print("\nEXPLICAÇÃO dos frames {}-{}".format(first, last))
    print("limiar do toque estrito {:.1f} m/s, do portão {:.1f} m/s; peso do toque {:.2f}, "
          "custo da troca {:.1f}".format(threshold, gate_threshold, TOUCH_WEIGHT, SWITCH_COST))
    print("\n  eventos (picos de dv):")
    print("  {:>7} {:>6} {:>5} {:>5} {:>5} {:>9} {:>9} {:>11} {:>11} {:>7} {:>7}  {}".format(
        "frame", "dv", "v1", "v2", "ang°", "dH inst", "dA inst", "dH depois", "dA depois",
        "dH ar", "dA ar", "virou"))
    for t in local_peaks(dv, [(0, max(0, first - 3), last + 4)], gate_threshold):
        if not first <= t <= last:
            continue
        s1, s2 = np.hypot(*v_in[t]), np.hypot(*v_out[t])
        ang = (np.degrees(np.arccos(np.clip(np.dot(v_in[t], v_out[t]) / (s1 * s2), -1, 1)))
               if s1 >= 3 and s2 >= 3 else np.nan)
        lo, hi = max(0, t - 2), t + 3
        end = min(segment_end[t] or len(dv), t + AFTER_FRAMES + 1)
        mh = np.nanmean(np.where(np.isfinite(d_home[t:end]), d_home[t:end], np.nan))
        ma = np.nanmean(np.where(np.isfinite(d_away[t:end]), d_away[t:end], np.nan))
        if t in strict:
            what = ("toque AÉREO de " if t in aerial else "toque estrito de ") + "HA"[strict[t]]
        elif in_flight(v_in[t], v_out[t]):
            what = "voo (ignorado)"
        elif dv[t] < threshold:
            what = "só portão (abaixo do limiar do toque)"
        else:
            what = "só portão (sem dono claro)"
        ah = "{:.1f}".format(air[0][lo:hi].min()) if air is not None else "-"
        aa = "{:.1f}".format(air[1][lo:hi].min()) if air is not None else "-"
        print("  {:>7} {:>6.1f} {:>5.1f} {:>5.1f} {:>5} {:>9.1f} {:>9.1f} {:>11.1f} {:>11.1f} {:>7} {:>7}  {}".format(
            t, dv[t], s1, s2, "-" if np.isnan(ang) else "{:.0f}".format(ang),
            d_home[lo:hi].min(), d_away[lo:hi].min(), mh, ma, ah, aa, what))
    print("\n  de 5 em 5 frames:")
    print("  {:>13} {:>5} {:>7} {:>7} {:>6} {:>6} {:>5} {:>9}".format(
        "frames", "prev", "ev H", "ev A", "dH", "dA", "bola", "liberada"))
    for a in range(first, last + 1, 5):
        b = min(last + 1, a + 5)
        dh, da = d_home[a:b], d_away[a:b]
        fin = np.isfinite(dh) & np.isfinite(da)
        lib = "".join(c for c, side in (("H", 0), ("A", 1))
                      if allow is not None and allow[side, a:b].any()) or "-"
        print("  {:>6}-{:<6} {:>5} {:>7.2f} {:>7.2f} {:>6} {:>6} {:>4.0%} {:>9}".format(
            a, b - 1, "".join(sorted(set(labels[a:b]))), ev[0, a:b].sum(), ev[1, a:b].sum(),
            "{:.1f}".format(dh[fin].mean()) if fin.any() else "-",
            "{:.1f}".format(da[fin].mean()) if fin.any() else "-", fin.mean(), lib))


def anchor_switches(piece, start, strict, permissive, window):
    """Move cada troca prevista para o frame do toque do novo time mais próximo (até ±window).

    O Viterbi decide SE a posse troca; o toque decide QUANDO. Sem isso, a proximidade de quem se
    aproxima da bola puxa a troca para antes do toque (as trocas certas do GT público saíam 19 a 23
    frames adiantadas), e o GT troca no toque (a 2-4 frames da recepção em 11351 e 15766).

    Preferência: o toque estrito do novo time; senão, um evento do portão que o libera. Uma troca
    só se move se continuar entre a troca anterior e a seguinte do trecho, sem cruzá-las.
    strict/permissive: arrays ordenados de frames por time. Devolve (rótulos, deslocamentos).
    """
    labels = piece.copy()
    changes = [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    moves = []
    new_pos = list(changes)
    for k, i in enumerate(changes):
        side = ro.HOME if labels[i] == "H" else ro.AWAY
        t = start + i
        target = None
        for frames in (strict[side], permissive[side]):
            near = frames[(frames >= t - window) & (frames <= t + window)]
            if len(near):
                target = int(near[np.argmin(np.abs(near - t))])
                break
        if target is None:
            continue
        j = target - start
        lo = new_pos[k - 1] + 1 if k > 0 else 1
        hi = changes[k + 1] - 1 if k + 1 < len(changes) else len(labels) - 1
        if lo <= j <= hi:
            new_pos[k] = j
            moves.append(j - i)
    # Reconstrói os rótulos a partir das novas posições das trocas.
    out = np.empty_like(labels)
    current, prev = labels[0], 0
    for k, j in enumerate(new_pos):
        out[prev:j] = current
        current = labels[changes[k]]
        prev = j
    out[prev:] = current
    return out, moves


def last_touch(touches, segments, n_frames):
    """Time do último toque, repetido para frente dentro de cada trecho; -1 antes do primeiro."""
    out = np.full(n_frames, -1, dtype=np.int8)
    marks = np.full(n_frames, -1, dtype=np.int8)
    for t, side, _ in touches:
        marks[t] = side
    for _, start, end in segments:
        current = -1
        for t in range(start, end):
            if marks[t] >= 0:
                current = marks[t]
            out[t] = current
    return out


# ------------------------------------------------------------------------------------------------
# Avaliação
# ------------------------------------------------------------------------------------------------

def changes(labels):
    """Frames em que o rótulo passa de H para A ou de A para H (ignora D e X)."""
    idx = np.flatnonzero(np.isin(labels, ["H", "A"]))
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1 and labels[a] != labels[b]:
            out.append((int(b), labels[b]))
    return out


def report_public(gt, labels, fps, allow=None):
    """spellF1 e macroF1 do avaliador oficial, e as trocas do GT ao lado das previstas."""
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels), bootstrap_n=0)
    print("\nGT público (avaliador oficial):")
    print("  spellF1 {:.4f}   macroF1 {:.4f}   posses previstas {} / GT {}".format(
        result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]))

    window = np.flatnonzero(gt != "X")
    a, b = window[0], window[-1] + 1
    gt_changes = changes(gt[a:b])
    pred_changes = changes(labels[a:b])
    pred_frames = np.array([f for f, _ in pred_changes]) if pred_changes else np.array([])
    print("\n  trocas no GT: {}   previstas: {}".format(len(gt_changes), len(pred_changes)))
    print("  {:>8} {:>4} {:>16} {:>10}".format("frame", "para", "prevista (desvio)",
                                               "liberada" if allow is not None else ""))
    used = set()
    for f, to in gt_changes:
        text = "não"
        if len(pred_frames):
            j = int(np.argmin(np.abs(pred_frames - f)))
            offset = int(pred_frames[j] - f)
            if abs(offset) <= 2 * fps and pred_changes[j][1] == to:  # até 2 s, mesmo sentido
                used.add(j)
                text = "{:+d} frames".format(offset)
        gate = ""
        if allow is not None:
            side = ro.HOME if to == "H" else ro.AWAY
            gate = "sim" if allow[side, a + f] else "NÃO"
        print("  {:>8} {:>4} {:>16} {:>10}".format(a + f, to, text, gate))
    spurious = [a + pred_changes[j][0] for j in range(len(pred_changes)) if j not in used]
    if spurious:
        print("  previstas sem par no GT (frames): {}".format(", ".join(map(str, spurious))))


# ------------------------------------------------------------------------------------------------

def main():
    global TAG_WEIGHT, TOUCH_WEIGHT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", required=True,
                        help="bola morta: frame,morta (analise/out/paradas_D.csv) ou frame,label com D")
    parser.add_argument("--live-ha", action="store_true",
                        help="só H e A contam como vivo (para usar o GT público como --dead)")
    parser.add_argument("--ball", help="CSV da bola própria; sem ele, usa a bola do ATD")
    parser.add_argument("--ball-cols", default="frame,x,y")
    parser.add_argument("--ball-units", choices=("norm", "m", "px"), default="norm",
                        help="norm = 0-1 como o ATD; m = metros; px = pixels do vídeo (ex.: a saída "
                             "do track_temporal.py), convertidos pela homografia")
    parser.add_argument("--ball-frame-base", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tag-weight", type=float, default=TAG_WEIGHT,
                        help="evidência por frame das fases do Smart Tagging (0 = desligado)")
    parser.add_argument("--no-gk-fix", action="store_true",
                        help="não corrige o time do goleiro (para comparar)")
    parser.add_argument("--no-despike", action="store_true",
                        help="não descarta as posições da bola que pulam (para comparar)")
    parser.add_argument("--explain", type=int, nargs=2, metavar=("INICIO", "FIM"),
                        help="explica frame a frame por que a posse trocou ou não nesse intervalo")
    parser.add_argument("--aerial", action="store_true",
                        help="toques no ar (cabeça, peito): câmera pela homografia e distância do "
                             "corpo de cada jogador ao raio da bola, quando ninguém está perto no chão")
    parser.add_argument("--touch-after", action="store_true",
                        help="atribui o toque a quem sai com a bola nos ~0,5 s seguintes "
                             "(experimental: não ajudou no GT público)")
    parser.add_argument("--prox-control", action="store_true",
                        help="a proximidade só conta com a bola sob controle, até CONTROL_SPEED "
                             "(experimental: no GT público atrasou trocas demais)")
    parser.add_argument("--anchor", action="store_true",
                        help="move as trocas para o toque do novo time mais próximo (experimental: "
                             "no GT público adiantou trocas que já saíam cedo)")
    parser.add_argument("--no-gate", action="store_true",
                        help="política antiga: sem o portão, toque com peso 1,5 (para comparar)")
    parser.add_argument("--touch-weight", type=float, default=None,
                        help="evidência de um toque estrito (padrão: 0,5 com o portão, 1,5 sem)")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "posse.csv"),
                        help="arquivo da submissão")
    parser.add_argument("--gt", default=os.path.join(ROOT, "evaluation", "gt",
                                                     "BAR-ATH_possession_gt_public.csv"))
    args = parser.parse_args()
    TAG_WEIGHT = args.tag_weight
    if args.touch_weight is not None:
        TOUCH_WEIGHT = args.touch_weight
    elif not args.no_gate:
        TOUCH_WEIGHT = TOUCH_WEIGHT_GATED

    n_frames = source_frames("tactical", GAME)
    fps = GAMES[GAME]["fps"]
    pitch = np.asarray(GAMES[GAME]["pitch"], dtype=float)
    blocks = half_video_blocks(GAME)

    dead = ro.read_dead(args.dead, n_frames, args.live_ha)
    segments = ro.live_segments(dead, blocks)
    print("trechos vivos: {}".format(len(segments)))

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = ro.team_of_each_track(meta_path)
    print("lendo o ATD inteiro ...")
    players, atd_ball = read_atd_players(raw_path, len(team), n_frames)

    # Goleiros: o mais recuado em cada grande área é do time que defende aquele gol.
    if args.no_gk_fix:
        teams = team
    else:
        teams, changed, keepers = goalkeeper_teams(players, team, blocks, pitch)
        print("goleiros: " + ("; ".join(
            "slot {} ({}, mais recuado do time em {:.0%} dos frames): time corrigido em {} frames"
            .format(slot + 1, "casa" if side == ro.HOME else "fora", frac, changed.get(slot, 0))
            for side, (slot, frac) in sorted(keepers.items()))
            or "nenhum slot claro (nenhum é o mais recuado em metade dos frames); nada corrigido"))

    if args.ball:
        ball = ro.read_ball(args.ball, n_frames, args.ball_cols.split(","), args.ball_frame_base)
        if args.ball_units == "m":
            ball = ball / pitch
        elif args.ball_units == "px":
            ball = pixels_to_field(ball)
        print("bola: {} ({} frames com bola)".format(
            os.path.basename(args.ball), int(np.isfinite(ball[:, 0]).sum())))
    else:
        ball = atd_ball
        print("bola: ATD")

    # Bola em metros, sem as posições que pulam para outro objeto. Sem interpolação: onde não há
    # bola, o Viterbi mantém a posse (decisão do tracking_bola/README.md, passo 15 e seção 6).
    raw_m = ball * pitch
    clean_m = raw_m.copy() if args.no_despike else despike(raw_m, SPIKE_SPAN, SPIKE_DIST)
    n_raw = int(np.isfinite(raw_m[:, 0]).sum())
    n_dropped = n_raw - int(np.isfinite(clean_m[:, 0]).sum())
    print("posições da bola descartadas por pular: {} de {} ({:.1%})".format(
        n_dropped, n_raw, n_dropped / max(1, n_raw)))

    # Dono no reinício: com a bola parada, a última posição vista vale por até 5 s (restart_owner.py).
    set_pieces = ro.read_set_pieces(require(["smart_tagging"])[0], fps)
    held = ro.hold_ball(clean_m / pitch, ro.LOOKBACK)
    view = PlayersView(players)
    restart_rows = [ro.decide(s, e, view, teams, held, tuple(pitch), set_pieces)
                    for _, s, e in segments]

    # Evidência por frame, em metros.
    ball_m = smooth(clean_m, SMOOTH)
    players_m = players * pitch.astype(np.float32)
    d_home, d_away = team_distances(players_m, ball_m, teams)
    dv, v_in, v_out = velocity_change(ball_m, segments, fps)

    air = None
    if args.aerial:
        centers, summary = estimate_camera(blocks, tuple(pitch), n_frames)
        if centers is None:
            print("toques aéreos desligados: " + summary)
        else:
            print("câmera recuperada da homografia: " + summary)
            air = air_distances(players_m, ball_m, teams, centers)

    jumpy = None
    if air is not None:  # posições que pulam, mesmo com --no-despike: não viram toque aéreo
        jumpy = np.isfinite(raw_m[:, 0]) & ~np.isfinite(despike(raw_m, SPIKE_SPAN, SPIKE_DIST)[:, 0])
    touches, threshold, noise, aerial = detect_touches(
        dv, v_in, v_out, d_home, d_away, segments, flight_filter=not args.no_gate,
        use_after=args.touch_after, air=air, jumpy=jumpy)
    print("ruído da bola: mudança de velocidade mediana {:.1f} m/s -> limiar de toque {:.1f} m/s"
          .format(noise, threshold))
    if air is not None:
        sides = [s for t, s, _ in touches if t in aerial]
        print("toques aéreos (ninguém a até {:.0f} m no chão, corpo a até {:.1f} m do raio da bola): "
              "{} (H {}, A {})".format(TOUCH_RADIUS, AIR_RADIUS, len(sides),
                                       sides.count(ro.HOME), sides.count(ro.AWAY)))
    allow = None
    if not args.no_gate:
        allow, gate = gate_mask(dv, v_in, v_out, d_home, d_away, ball_m, segments, noise, air,
                                jumpy)
        if air is not None:
            print("portão: {} liberações pelo raio da bola (bola no ar)".format(gate["aereos"]))
        live_mask = np.zeros(n_frames, dtype=bool)
        for _, s, e in segments:
            live_mask[s:e] = True
        print("portão: {} eventos (limiar {:.1f} m/s; {} descartados como voo), {} reaparecimentos; "
              "troca liberada em {:.0%} dos frames vivos para H e {:.0%} para A".format(
                  gate["eventos"], gate["limiar"], gate["voo"], gate["reaparecimentos"],
                  allow[ro.HOME, live_mask].mean(), allow[ro.AWAY, live_mask].mean()))
    speed = ball_speed(ball_m, fps) if args.prox_control else None
    ev = proximity_evidence(d_home, d_away, speed)
    if speed is not None:
        live_idx = np.concatenate([np.arange(s, e) for _, s, e in segments]) if segments else []
        known = np.isfinite(speed[live_idx])
        print("proximidade só com a bola sob controle (até {:.0f} m/s): {:.0%} dos frames vivos "
              "com velocidade medida".format(CONTROL_SPEED,
                                             np.mean(speed[live_idx][known] <= CONTROL_SPEED)
                                             if known.any() else 0.0))
    for t, side, _ in touches:
        ev[side, t] += TOUCH_WEIGHT
    if TAG_WEIGHT > 0:
        ev += tag_evidence(require(["smart_tagging"])[0], n_frames, fps)
    print("toques detectados: {}  (H {}, A {})".format(
        len(touches), sum(s == ro.HOME for _, s, _ in touches),
        sum(s == ro.AWAY for _, s, _ in touches)))

    live_labels = predict(segments, ev, last_touch(touches, segments, n_frames), restart_rows, allow)

    if allow is not None and args.anchor:
        strict = [np.array(sorted(t for t, s, _ in touches if s == side), dtype=int)
                  for side in (ro.HOME, ro.AWAY)]
        permissive = [np.array(sorted(gate["frames"][side]), dtype=int)
                      for side in (ro.HOME, ro.AWAY)]
        all_moves = []
        for (start, end), piece in list(live_labels.items()):
            live_labels[(start, end)], moves = anchor_switches(piece, start, strict, permissive,
                                                               GATE_WINDOW)
            all_moves += moves
        if all_moves:
            print("ancoragem no toque: {} trocas movidas, deslocamento mediano {:+.0f} frames "
                  "(de {:+d} a {:+d})".format(len(all_moves), np.median(all_moves),
                                              min(all_moves), max(all_moves)))

    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        labels[first:last + 1] = "D"
    for (start, end), piece in live_labels.items():
        labels[start:end] = piece
    ro.write_submission(args.out, labels)


    n_changes = len(changes(labels))
    live = int(sum(e - s for _, s, e in segments))
    print("trocas de posse previstas: {}   posses ~{}   duração média {:.1f} s".format(
        n_changes, n_changes + len(segments), live / fps / max(1, n_changes + len(segments))))
    print("submissão: {}".format(args.out))

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    touch_path = os.path.join(OUT_DIR, "toques.csv")
    pd.DataFrame([{"frame": t, "track_id": "toque", "team": "HA"[side],
                   "x": ball_m[t, 0] / pitch[0], "y": ball_m[t, 1] / pitch[1]}
                  for t, side, _ in touches]).to_csv(touch_path, index=False)
    print("camada de toques para o viewer: {}".format(touch_path))

    if os.path.isfile(args.gt):
        report_public(ro.read_labels(args.gt, n_frames), labels, fps, allow)

    if args.explain:
        gate_threshold = gate["limiar"] if allow is not None else threshold
        explain(args.explain[0], args.explain[1], dv, v_in, v_out, d_home, d_away, ev, touches,
                allow, labels, threshold, gate_threshold, segments, air, aerial)


if __name__ == "__main__":
    main()