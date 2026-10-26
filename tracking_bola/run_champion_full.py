"""Modelo final de rastreamento da bola em TODOS os frames avaliados: Roboflow -> máscara do campo -> fusão com o ATD.

Colocar em tracking_bola/, junto de detect_yolo.py, mask_pitch.py, fuse_atd_wasb.py, compare_trackers.py,
eval_atd.py e check_projection.py (reaproveita o código deles). Precisa de ultralytics e opencv-python:

    python tracking_bola/run_champion_full.py --limit-frames 200     # teste de tempo: só 200 frames
    python tracking_bola/run_champion_full.py                        # jogo todo (retoma de onde parou)

O que faz, com os parâmetros escolhidos nos frames anotados:
  1. DETECÇÃO (lenta, com salvamento e retomada): o detector de bola do Roboflow (YOLOv8, classe 0) em
     ladrilhos de 640x640 com 20% de sobreposição, em cada frame dos dois tempos avaliados
     (games.half_video_blocks: 158.857 frames). Todos os candidatos com conf >= --conf vão para
     out/full_rf.csv (frame,x,y,conf). O vídeo é lido uma vez, em ordem.
  2. MÁSCARA + FUSÃO (rápida, roda sozinha quando a detecção termina): descarta candidatos a mais de
     --margin metros fora do campo (homografia do frame) e funde com o ATD pela regra do fuse_atd_wasb.py
     (--solo-conf 0.8, --override-conf 0.8, --radius 40, --floor 0.2). Saídas:
       out/full_rf_campo.csv   candidatos depois da máscara (frame,x,y,conf)
       out/full_bola.csv       a bola de cada frame avaliado com bola (frame,x,y,regra), em pixels do vídeo
     Frame avaliado sem linha em full_bola.csv = sem bola. --post-only refaz só esta etapa (por exemplo,
     com outra margem) sem detectar de novo.

Salvamento: a cada --save-every frames (padrão 500 = 20 s de vídeo) o CSV é gravado e o arquivo
out/full_rf.csv.progress marca até onde foi. Ctrl+C, queda de energia ou fechar a janela perdem no
máximo esse trecho: rodar o MESMO comando de novo continua dali. Se os parâmetros da detecção mudarem,
ele recusa continuar; use --restart para recomeçar do zero.

ESTES ARQUIVOS SÃO DADOS DERIVADOS DO VÍDEO: ficam em out/, que não vai para o Git.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402
from check_projection import read_ball  # noqa: E402
from eval_atd import atd_pixel  # noqa: E402
from compare_trackers import load_tracker  # noqa: E402
from fuse_atd_wasb import fuse_frame  # noqa: E402
from mask_pitch import inside, pitch_size  # noqa: E402
import detect_yolo  # noqa: E402

OUT = os.path.join(HERE, "out")
DEFAULT_WEIGHTS = os.path.join(HERE, "pesos", "football-ball-detection.pt")
COLUMNS = ["frame", "x", "y", "conf"]


def graded_frames(start, end):
    """Frames dos dois tempos avaliados, dentro de [start, end], em ordem."""
    frames = []
    for _, first, last in half_video_blocks(GAME):
        frames.extend(range(max(first, start), min(last, end) + 1))
    return frames


def write_progress(path, done, signature):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump({"done": done, "signature": signature}, handle)
    os.replace(temporary, path)


def trim_csv(path, first_bad_frame):
    """Apaga linhas com frame >= first_bad_frame (sobras de um trecho não confirmado)."""
    if not os.path.isfile(path):
        return
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        kept = [row for row in reader if row and int(row[0]) < first_bad_frame]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(kept)


def detect(args, frames, model_factory):
    """Etapa 1. Devolve True se terminou todos os frames."""
    import cv2
    signature = {"weights": os.path.basename(args.weights), "conf": args.conf, "tile": args.tile,
                 "overlap": args.overlap, "imgsz": args.imgsz, "class_id": args.class_id, "merge": args.merge,
                 "frames": len(frames), "first": frames[0], "last": frames[-1]}
    progress = args.raw + ".progress"
    if args.restart:
        for path in (args.raw, progress):
            if os.path.isfile(path):
                os.remove(path)
    done = 0
    if os.path.isfile(progress):
        with open(progress) as handle:
            saved = json.load(handle)
        if saved["signature"] != signature:
            sys.exit("{} é de uma execução com outros parâmetros ({}).\nUse --restart para recomeçar.".format(
                progress, saved["signature"]))
        done = int(saved["done"])
        trim_csv(args.raw, frames[done] if done < len(frames) else frames[-1] + 1)
        if done:
            print("retomando: {} de {} frames já feitos".format(done, len(frames)))
    elif os.path.isfile(args.raw):
        sys.exit("{} já existe e não há {}. Use --restart para sobrescrever.".format(args.raw, progress))
    if done >= len(frames):
        print("detecção já completa ({} frames)".format(len(frames)))
        return True
    if not os.path.isfile(args.raw):
        with open(args.raw, "w", newline="") as handle:
            csv.writer(handle).writerow(COLUMNS)
        # grava o progresso já no começo: se algo falhar antes do primeiro frame (pesos, vídeo),
        # a próxima execução retoma do zero em vez de achar um CSV "órfão"
        write_progress(progress, 0, signature)

    model = model_factory(args.weights)
    names = getattr(model, "names", None) or {}
    print("classe {} do modelo: '{}'".format(args.class_id, names.get(args.class_id)))
    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")

    limit = len(frames) if args.limit_frames is None else min(len(frames), done + args.limit_frames)
    print("{} frames avaliados; processando de {} a {}; salva a cada {} frames".format(
        len(frames), done, limit, args.save_every))
    handle = open(args.raw, "a", newline="")
    writer = csv.writer(handle)
    pending, position, started, started_done = [], None, time.time(), done

    def flush(upto):
        writer.writerows(pending)
        handle.flush()
        os.fsync(handle.fileno())
        write_progress(progress, upto, signature)
        del pending[:]

    try:
        while done < limit:
            frame = frames[done]
            if position != frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                sys.exit("não consegui ler o frame {}".format(frame))
            position = frame + 1
            for x, y, conf in sorted(detect_yolo.detect_frame(model, image, args), key=lambda c: -c[2]):
                pending.append((frame, "{:.2f}".format(x), "{:.2f}".format(y), "{:.4f}".format(conf)))
            done += 1
            if done % args.save_every == 0:
                flush(done)
            if (done - started_done) % args.report_every == 0:
                per_frame = (time.time() - started) / (done - started_done)
                print("  {}/{} frames, {:.2f} s/frame, faltam ~{:.1f} h".format(
                    done, len(frames), per_frame, per_frame * (len(frames) - done) / 3600.0))
    except KeyboardInterrupt:
        print("\ninterrompido: salvando o que já foi feito.")
    finally:
        flush(done)
        handle.close()
    if done > started_done:
        per_frame = (time.time() - started) / (done - started_done)
        print("{} frames nesta execução, {:.2f} s/frame; jogo todo ~{:.1f} h nesta máquina.".format(
            done - started_done, per_frame, per_frame * len(frames) / 3600.0))
    print("detecção: {}/{} frames. {}".format(done, len(frames), "" if done == len(frames)
                                               else "Rode o mesmo comando de novo para continuar."))
    return done == len(frames)


def post(args, frames):
    """Etapa 2: máscara do campo + fusão com o ATD, em todos os frames avaliados."""
    _, _, every = load_tracker(args.raw)
    n_frames = source_frames("tactical", GAME)
    matrices, _, valid = load_homography()
    ball = read_ball(require(["atd_raw"])[0], n_frames)
    length, width = pitch_size()
    margin_u, margin_v = args.margin / length, args.margin / width

    kept_rows, fused_rows, counts, removed = [], [], {}, 0
    for frame in frames:
        candidates = every.get(frame, [])
        if candidates and valid[frame]:
            inverse = np.linalg.inv(matrices[frame])
            masked = []
            for x, y, conf in candidates:
                u, v = project(inverse, x, y)
                if inside(float(u), float(v), margin_u, margin_v):
                    masked.append((x, y, conf))
                else:
                    removed += 1
            candidates = masked
        for x, y, conf in candidates:
            kept_rows.append((frame, "{:.2f}".format(x), "{:.2f}".format(y), "{:.4f}".format(conf)))
        point, rule = fuse_frame(atd_pixel(ball, matrices, valid, frame), candidates,
                                 args.radius, args.override_conf, args.solo_conf, args.floor)
        counts[rule] = counts.get(rule, 0) + 1
        if point is not None:
            fused_rows.append((frame, "{:.2f}".format(point[0]), "{:.2f}".format(point[1]), rule))

    with open(args.masked, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(kept_rows)
    with open(args.final, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x", "y", "regra"])
        writer.writerows(fused_rows)

    print("\nmáscara de {:g} m: {} candidatos removidos fora do campo".format(args.margin, removed))
    print("fusão em {} frames avaliados; com bola: {} ({:.1f}%)".format(
        len(frames), len(fused_rows), 100.0 * len(fused_rows) / max(1, len(frames))))
    for rule in sorted(counts):
        print("  {:<34} {:7d}".format(rule, counts[rule]))
    print("saídas: {}\n        {}".format(args.masked, args.final))


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="pesos do Roboflow (padrão pesos/football-ball-detection.pt)")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.2, help="conf mínima guardada (padrão 0.2 = o piso da fusão)")
    parser.add_argument("--merge", type=float, default=6.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--margin", type=float, default=2.0, help="máscara: metros fora das linhas (padrão 2)")
    parser.add_argument("--solo-conf", type=float, default=0.8)
    parser.add_argument("--override-conf", type=float, default=0.8)
    parser.add_argument("--radius", type=float, default=40.0)
    parser.add_argument("--floor", type=float, default=0.2)
    parser.add_argument("--start", type=int, default=0, help="primeiro frame (padrão: início do 1º tempo)")
    parser.add_argument("--end", type=int, default=None, help="último frame (padrão: fim do 2º tempo)")
    parser.add_argument("--save-every", type=int, default=500, help="salva a cada N frames (padrão 500)")
    parser.add_argument("--report-every", type=int, default=100, help="mostra o andamento a cada N frames")
    parser.add_argument("--limit-frames", type=int, default=None, help="só N frames nesta execução (teste de tempo)")
    parser.add_argument("--restart", action="store_true", help="apaga a detecção feita e recomeça")
    parser.add_argument("--post-only", action="store_true", help="refaz só máscara + fusão")
    parser.add_argument("--raw", default=os.path.join(OUT, "full_rf.csv"))
    parser.add_argument("--masked", default=os.path.join(OUT, "full_rf_campo.csv"))
    parser.add_argument("--final", default=os.path.join(OUT, "full_bola.csv"))
    args = parser.parse_args(argv)
    args.imgsz = args.imgsz or args.tile
    args.stride = max(1, int(args.tile * (1 - args.overlap)))

    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit("opencv não está instalado:  pip install opencv-python")
    for path in (args.raw, args.masked, args.final):
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
    n_frames = source_frames("tactical", GAME)
    frames = graded_frames(args.start, n_frames - 1 if args.end is None else args.end)
    if not frames:
        sys.exit("nenhum frame avaliado no intervalo pedido")

    if not args.post_only:
        if model_factory is None:
            if not os.path.isfile(args.weights):
                sys.exit("não achei os pesos em {}. Veja o examples/soccer/setup.sh do roboflow/sports.".format(args.weights))
            try:
                from ultralytics import YOLO as model_factory
            except ImportError:
                sys.exit("ultralytics não está instalado:  pip install ultralytics")
        if not detect(args, frames, model_factory):
            return
    else:
        if not os.path.isfile(args.raw):
            sys.exit("não achei {}: rode a detecção antes".format(args.raw))
        progress = args.raw + ".progress"
        if os.path.isfile(progress):
            with open(progress) as handle:
                done = int(json.load(handle)["done"])
            if done < len(frames):
                print("ATENÇÃO: a detecção só cobriu {} de {} frames; o resto sai só com o ATD.".format(done, len(frames)))
    post(args, frames)


if __name__ == "__main__":
    main()