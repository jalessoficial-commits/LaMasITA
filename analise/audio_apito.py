"""Apito do árbitro pelo áudio do vídeo tático (M1 do detector de paradas).

    python analise/audio_apito.py            # ~2-3 min; precisa do ffmpeg no PATH

O vídeo tático traz o som do estádio (AAC, 48 kHz) no mesmo relógio dos frames. O apito de um
árbitro é um tom QUASE PURO e de frequência fixa (neste jogo ~3,8 kHz, com harmônico em ~4,3 kHz);
a torcida e o vento são banda larga, e os apitos/cantos da torcida variam de frequência. Por isso o
detector não procura "som agudo", e sim energia numa banda estreita muito acima das bandas vizinhas:

    snr(t) = max_dB[f0 - 125, f0 + 125] - max(max_dB[f0 - 450, f0 - 150], max_dB[f0 + 150, f0 + 400])

f0 é calibrado nas SAÍDAS DE BOLA do Smart Tagging (início dos tempos e depois de gol): o árbitro
sempre apita ali. Em cada janela [start_frame - 8 s, start_frame + 8 s] pega-se a frequência do tom
estreito mais forte; f0 é a mediana. (Contar tons no jogo inteiro não funciona: cantos e vozes da
torcida geram mais tons estáveis do que o apito.)

Saídas (fora do git: derivam dos dados restritos):

    viewer/data/apito_tactical.bin    float32 (n_frames,): snr em dB por frame (máximo no frame)
    viewer/data/apito_tactical.json   f0, bandas, limiar, eventos
    analise/out/apitos.csv            frame, t_s, dur_s, snr_max_db

    from audio_apito import load_apito
    snr, meta = load_apito()          # meta["eventos"] = [[frame_inicio, frame_fim, snr_max], ...]
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

FPS = GAMES[GAME]["fps"]
N_FRAMES = GAMES[GAME]["n_video_frames"]
DATA_OUT = os.path.join(ROOT, "viewer", "data")
OUT_DIR = os.path.join(HERE, "out")

# Params
SR = 16000                 # Hz; o apito (2-5 kHz) cabe com folga abaixo de 8 kHz
NFFT, HOP = 2048, 512      # resolução 7,8 Hz; passo 32 ms
CHUNK_S = 120              # processa o áudio em blocos de 2 min (memória constante)
SEARCH_BAND = (2000, 5000)
CORE_HALF = 125            # Hz em volta de f0
GUARD, SIDE = 150, 150     # bandas vizinhas: [f0 - CORE - GUARD - SIDE, f0 - CORE - GUARD] e simétrica;
                           # estreitas de propósito: o apito tem um 2º tom ~+500 Hz que não pode cair nelas
THRESHOLD_DB = 10.0
MIN_DUR_S = 0.12
MERGE_GAP_S = 1.0          # apitos repetidos (prrr-prrr) viram um evento


def find_video():
    path = asset_paths(GAME)["video"]
    if os.path.exists(path):
        return path
    mp4 = sorted(glob.glob(os.path.join(os.path.dirname(path), "*.mp4")))
    if not mp4:
        raise SystemExit("nenhum .mp4 em {}".format(os.path.dirname(path)))
    return mp4[0]                      # mesma regra do viewer: o primeiro .mp4 da pasta


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg não encontrado (instale ou pip install imageio-ffmpeg)")


def audio_chunks(video):
    """Gera blocos float32 mono a SR Hz, lidos de um pipe do ffmpeg (nada é gravado em disco)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = CHUNK_S * SR * 2
    while True:
        raw = proc.stdout.read(size)
        if not raw:
            break
        yield np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    proc.wait()


def spectrogram_db(x):
    """(freqs, tempos relativos ao início do bloco, dB) com janela de Hann."""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) < NFFT:
        x = np.pad(x, (0, NFFT - len(x)))
    frames = sliding_window_view(x, NFFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * np.hanning(NFFT), axis=1)) ** 2
    f = np.fft.rfftfreq(NFFT, 1.0 / SR)
    t = (np.arange(len(frames)) * HOP + NFFT / 2) / SR
    return f, t, 10 * np.log10(spec.T + 1e-14)


def band_max(f, L, lo, hi):
    return L[(f >= lo) & (f <= hi)].max(0)


def spectra(video):
    """Gera (f, t_absoluto, dB) bloco a bloco, com sobreposição de NFFT amostras entre blocos."""
    offset, carry = 0.0, np.zeros(0, np.float32)
    for chunk in audio_chunks(video):
        x = np.concatenate([carry, chunk])
        f, t, L = spectrogram_db(x)
        t_abs = t + offset - len(carry) / SR
        # descarta as colunas que já saíram no bloco anterior (a última saiu em ~offset - NFFT/2)
        keep = t_abs > offset - 0.5 * NFFT / SR if len(carry) else np.ones_like(t_abs, bool)
        yield f, t_abs[keep], L[:, keep]
        carry = x[-NFFT:]
        offset += len(chunk) / SR


def narrow_peaks(f, L):
    """Por coluna: frequência do pico em 2-5 kHz e o quanto ele se destaca das bandas a ±150-450 Hz."""
    from scipy.ndimage import maximum_filter1d
    df = f[1] - f[0]
    o1, o2 = int(round(GUARD / df)), int(round((GUARD + SIDE) / df))
    width = o2 - o1
    # side_max[i] = max de L em [i + o1, i + o2) e em (i - o2, i - o1]
    mf = maximum_filter1d(L, size=width, axis=0, origin=-(width // 2))      # janela [i, i + width)
    up = np.full_like(L, -200.0); up[:-o1] = mf[o1:]
    mb = maximum_filter1d(L, size=width, axis=0, origin=(width - 1) // 2)  # janela (i - width, i]
    dn = np.full_like(L, -200.0); dn[o1:] = mb[:-o1]
    band = np.flatnonzero((f >= SEARCH_BAND[0]) & (f <= SEARCH_BAND[1]))
    k = band[L[band].argmax(0)]
    cols = np.arange(L.shape[1])
    return f[k], L[k, cols] - np.maximum(up[k, cols], dn[k, cols])


def calibrate_f0(pf, nsnr, t, anchors_s, half_window=8.0, default=3820.0):
    """Mediana da frequência do tom estreito mais forte em cada janela de saída de bola."""
    picks = []
    for a in anchors_s:
        sel = np.flatnonzero((t >= a - half_window) & (t <= a + half_window))
        if len(sel) and nsnr[sel].max() > 12:
            picks.append(float(pf[sel[np.argmax(nsnr[sel])]]))
    if len(picks) < 2:
        return default, picks
    return float(np.median(picks)), picks


def run_calibration(video, t0):
    """Passada 1: calibra f0 nas saídas de bola (guarda 2 números por coluna)."""
    pf, nsnr, tt = [], [], []
    for f, t, L in spectra(video):
        a_, b_ = narrow_peaks(f, L)
        pf.append(a_); nsnr.append(b_); tt.append(t)
    pf, nsnr, tt = map(np.concatenate, (pf, nsnr, tt))
    anchors = kickoff_anchors_s()
    f0, picks = calibrate_f0(pf, nsnr, tt, anchors)
    print("  passada 1 em {:.0f} s; f0 = {:.0f} Hz (saídas de bola: {} -> {})".format(
        time.time() - t0, f0, len(anchors), [round(p) for p in picks]))
    return f0, picks


def kickoff_anchors_s():
    path = os.path.join(DATA_OUT, "smart_tagging.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as h:
        events = json.load(h)
    return [e["start_frame"] / FPS for e in events
            if (e.get("code") or "").upper() == "SET PIECES" and "kick off" in (e.get("type") or "").lower()]


def whistle_snr(f, L, f0):
    core = band_max(f, L, f0 - CORE_HALF, f0 + CORE_HALF)
    lo = band_max(f, L, f0 - CORE_HALF - GUARD - SIDE, f0 - CORE_HALF - GUARD)
    hi = band_max(f, L, f0 + CORE_HALF + GUARD, f0 + CORE_HALF + GUARD + SIDE)
    return core - np.maximum(lo, hi)


def events_from(snr, t, threshold=THRESHOLD_DB, min_dur=MIN_DUR_S, merge_gap=MERGE_GAP_S):
    on = snr > threshold
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    raw = [(t[s], t[e - 1], float(snr[s:e].max())) for s, e in zip(edges[::2], edges[1::2])
           if t[e - 1] - t[s] >= min_dur]
    merged = []
    for s, e, m in raw:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e, max(merged[-1][2], m))
        else:
            merged.append((s, e, m))
    return merged


def per_frame(snr, t, n_frames=N_FRAMES):
    """Máximo do snr dentro de cada frame de vídeo (1/25 s)."""
    fr = np.clip((t * FPS).astype(int), 0, n_frames - 1)
    out = np.full(n_frames, -40.0, np.float32)
    np.maximum.at(out, fr, snr.astype(np.float32))
    return out


def load_apito():
    with open(os.path.join(DATA_OUT, "apito_tactical.json"), encoding="utf-8") as h:
        meta = json.load(h)
    snr = np.fromfile(os.path.join(DATA_OUT, "apito_tactical.bin"), dtype=np.float32)
    return snr, meta


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--f0", type=float, default=None, help="pula a calibração e usa este f0 (Hz)")
    args = parser.parse_args()
    t0 = time.time()
    video = find_video()
    print("áudio de", video)
    picks = []
    if args.f0:
        f0 = args.f0
    else:
        f0, picks = run_calibration(video, t0)
    # passada 2: snr na banda do apito
    snr, t = [], []
    for f, t_, L in spectra(video):
        snr.append(whistle_snr(f, L, f0)); t.append(t_)
    snr, t = np.concatenate(snr), np.concatenate(t)
    print("  passada 2 ({:.0f} s de áudio) em {:.0f} s".format(t[-1], time.time() - t0))
    ev = events_from(snr, t)
    snr_frame = per_frame(snr, t)
    os.makedirs(OUT_DIR, exist_ok=True)
    snr_frame.tofile(os.path.join(DATA_OUT, "apito_tactical.bin"))
    eventos = [[int(round(s * FPS)), int(round(e * FPS)), round(m, 1)] for s, e, m in ev]
    meta = {"game": GAME, "fps": FPS, "n_frames": N_FRAMES, "f0_hz": f0, "core_half_hz": CORE_HALF,
            "guard_hz": GUARD, "side_hz": SIDE, "threshold_db": THRESHOLD_DB, "min_dur_s": MIN_DUR_S,
            "merge_gap_s": MERGE_GAP_S, "eventos": eventos, "f0_por_saida_de_bola_hz": picks}
    with open(os.path.join(DATA_OUT, "apito_tactical.json"), "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=1)
    with open(os.path.join(OUT_DIR, "apitos.csv"), "w", encoding="utf-8") as h:
        h.write("frame,t_s,dur_s,snr_max_db\n")
        for s, e, m in ev:
            h.write("{},{:.2f},{:.2f},{:.1f}\n".format(int(round(s * FPS)), s, e - s, m))
    print("  {} apitos -> {}, {}".format(len(ev), os.path.join(DATA_OUT, "apito_tactical.*"),
                                         os.path.join(OUT_DIR, "apitos.csv")))
    print("  janela do GT (440-760 s):", ["{:.1f}s".format(s) for s, e, m in ev if 440 <= s <= 760])
    print("  total {:.0f} s".format(time.time() - t0))


if __name__ == "__main__":
    main()
