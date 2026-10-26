"""WASB (escala escolhida) em TODOS os frames avaliados do jogo, com salvamento e retomada.

Colocar em tracking_bola/, junto de detect_wasb.py (este script reaproveita o código dele). Mesmos
argumentos de modelo do detect_wasb.py:

    python tracking_bola/run_wasb_full.py --wasb-dir tracking_bola/third_party/WASB-SBDT --weights tracking_bola/pesos/wasb_soccer_best.pth.tar --scale 0.6
    python tracking_bola/run_wasb_full.py ... --limit-jobs 20      # teste de tempo: só 20 passadas

Quais frames: os dois tempos de jogo que a competição avalia, em `games.half_video_blocks`
(1º tempo 11.185-90.651 e 2º tempo 110.808-190.197: 158.857 frames). Antes do jogo, intervalo e
depois do apito final não são avaliados, então nada é processado ali. --all-frames processa o vídeo
inteiro (199.319 frames).

Como é rápido o bastante: o modelo recebe 3 frames consecutivos e devolve o mapa de calor dos 3. Aqui
o vídeo é lido uma vez, em ordem, em trios SEM sobreposição (passo 3), e os 3 mapas são aproveitados:
uma passada serve a 3 frames, 3x menos passadas que no detect_wasb.py (que processa 1 frame por trio).
O custo é que o frame da ponta do trio vê o passado ou o futuro só de um lado. No artigo do WASB, o passo 3
perde pouco em relação ao passo 1 (AP 83,6 contra 86,2). Para medir no SEU vídeo, rode o
detect_wasb.py com --position 0 e --position 2 nos frames anotados e compare com --position 1.

Saída: tracking_bola/out/wasb_full.csv com frame,x,y,conf (vários candidatos por frame, como no
detect_wasb.py), em pixels do vídeo original. Frame sem linha = nenhum pico acima do limiar.
ESSE ARQUIVO É DADO DERIVADO DO VÍDEO: fica em out/, que não vai para o Git.

Salvamento e retomada: a cada --save-every passadas o script grava o CSV e um arquivo
<saída>.progress com quantas passadas já terminaram. Ctrl+C, queda de energia ou fechar a janela não
perdem o trabalho: rodar o MESMO comando de novo continua de onde parou. Se os parâmetros mudarem
(escala, limiar, intervalo), ele recusa continuar; use --restart para recomeçar do zero.

Dividir o trabalho (dois computadores, ou dois horários): --start N e --end M limitam o intervalo de
frames; use --out diferentes e junte depois com --merge:

    python tracking_bola/run_wasb_full.py ... --start 11185  --end 90651  --out tracking_bola/out/wasb_h1.csv
    python tracking_bola/run_wasb_full.py ... --start 110808 --end 190197 --out tracking_bola/out/wasb_h2.csv
    python tracking_bola/run_wasb_full.py --merge tracking_bola/out/wasb_h1.csv tracking_bola/out/wasb_h2.csv --out tracking_bola/out/wasb_full.csv

Tempo: o script imprime o tempo por passada e a estimativa de horas restantes. Em CPU pode levar
muitas horas; deixe o computador ligado e sem suspender. Rode primeiro com --limit-jobs 20 para
ver o tempo antes de se comprometer.
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
from games import GAME, half_video_blocks, require, source_frames  # noqa: E402
import detect_wasb as dw  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "out", "wasb_full.csv")
COLUMNS = ["frame", "x", "y", "conf"]


def build_jobs(blocks, start, end):
    """Lista de passadas [(frame inicial do trio, [posições 0..2 que valem])], em ordem de frame.
    Os blocos são cortados pelo intervalo [start, end]. A última passada de um bloco pode ser deslocada
    para trás para caber (e então só vale nos frames que sobraram)."""
    jobs = []
    for first, last in blocks:
        first, last = max(first, start), min(last, end)
        if first > last:
            continue
        frame = first
        while frame <= last:
            if frame + dw.FRAMES_IN - 1 <= last:
                jobs.append((frame, [0, 1, 2]))
                frame += dw.FRAMES_IN
            else:
                begin = max(0, last - dw.FRAMES_IN + 1)
                wanted = [i for i in range(dw.FRAMES_IN) if frame <= begin + i <= last]
                jobs.append((begin, wanted))
                break
    return jobs


def trim_csv(path, first_bad_frame):
    """Apaga do CSV as linhas com frame >= first_bad_frame (sobras de uma passada que não foi confirmada)."""
    if not os.path.isfile(path):
        return
    kept = []
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row and int(row[0]) < first_bad_frame:
                kept.append(row)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(kept)


def write_progress(path, done, signature):
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump({"done": done, "signature": signature}, handle)
    os.replace(temporary, path)


def merge_files(paths, out):
    rows = set()
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if row:
                    rows.add((int(row[0]), row[1], row[2], row[3]))
    folder = os.path.dirname(out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for frame, x, y, conf in sorted(rows, key=lambda r: (r[0], -float(r[3]))):
            writer.writerow([frame, x, y, conf])
    print("{} linhas gravadas em {}".format(len(rows), out))


def main(argv=None, model_factory=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wasb-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--scale", type=float, default=0.6, help="escala do frame (padrão 0.6)")
    parser.add_argument("--score-threshold", type=float, default=0.2, help="limiar do mapa de calor (padrão 0.2)")
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--batch", type=int, default=8, help="ladrilhos por passada do modelo (padrão 8)")
    parser.add_argument("--merge-px", dest="merge", type=float, default=6.0, help="junta candidatos a menos disso (px)")
    parser.add_argument("--bgr", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--start", type=int, default=0, help="primeiro frame a processar")
    parser.add_argument("--end", type=int, default=None, help="último frame a processar")
    parser.add_argument("--all-frames", action="store_true", help="processa o vídeo inteiro, não só os dois tempos")
    parser.add_argument("--save-every", type=int, default=50, help="salva a cada N passadas (padrão 50)")
    parser.add_argument("--report-every", type=int, default=20, help="imprime o andamento a cada N passadas")
    parser.add_argument("--limit-jobs", type=int, default=None, help="só N passadas (teste de tempo)")
    parser.add_argument("--restart", action="store_true", help="apaga o que já foi feito e recomeça")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--merge", dest="merge_files", nargs="+", default=None, metavar="CSV",
                        help="só junta estes CSVs em --out e sai")
    args = parser.parse_args(argv)

    if args.merge_files:
        merge_files(args.merge_files, args.out)
        return
    if model_factory is None and not args.wasb_dir:
        parser.error("--wasb-dir é obrigatório")
    try:
        import cv2
    except ImportError:
        sys.exit("opencv não está instalado:  python -m pip install opencv-python")

    n_frames = source_frames("tactical", GAME)
    blocks = [(0, n_frames - 1)] if args.all_frames else [(first, last) for _, first, last in half_video_blocks(GAME)]
    end = n_frames - 1 if args.end is None else min(args.end, n_frames - 1)
    jobs = build_jobs(blocks, args.start, end)
    if not jobs:
        sys.exit("nenhum frame no intervalo pedido")
    signature = {"scale": args.scale, "threshold": args.score_threshold, "overlap": args.tile_overlap,
                 "merge": args.merge, "bgr": args.bgr, "jobs": len(jobs), "first": jobs[0][0], "last_job": jobs[-1][0]}

    progress_path = args.out + ".progress"
    done = 0
    if args.restart:
        for path in (args.out, progress_path):
            if os.path.isfile(path):
                os.remove(path)
    if os.path.isfile(progress_path):
        with open(progress_path) as handle:
            saved = json.load(handle)
        if saved["signature"] != signature:
            sys.exit("o {} é de uma execução com parâmetros diferentes ({}). Use --restart para recomeçar, ou\n"
                     "apague-o se quiser outro --out.".format(progress_path, saved["signature"]))
        done = int(saved["done"])
        if done:
            trim_csv(args.out, jobs[done][0] + jobs[done][1][0] if done < len(jobs) else n_frames)
            print("retomando: {} de {} passadas já feitas".format(done, len(jobs)))
    elif os.path.isfile(args.out):
        sys.exit("{} já existe e não há {}. Use --restart para sobrescrever.".format(args.out, progress_path))
    folder = os.path.dirname(args.out)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    if not os.path.isfile(args.out):
        with open(args.out, "w", newline="") as handle:
            csv.writer(handle).writerow(COLUMNS)

    predict, args.input_wh = (model_factory or dw.load_real_model)(args)
    capture = cv2.VideoCapture(require(["video"])[0])
    if not capture.isOpened():
        sys.exit("não consegui abrir o vídeo tactical")

    total = len(jobs)
    limit = total if args.limit_jobs is None else min(total, done + args.limit_jobs)
    print("{} passadas no total ({} frames); processando de {} a {}".format(
        total, sum(len(w) for _, w in jobs), done, limit))
    pending, position, seek_warnings = [], None, 0
    started, started_done = time.time(), done
    handle = open(args.out, "a", newline="")
    writer = csv.writer(handle)

    def flush(upto):
        for row in pending:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        write_progress(progress_path, upto, signature)
        del pending[:]

    try:
        while done < limit:
            begin, wanted = jobs[done]
            if position != begin:
                capture.set(cv2.CAP_PROP_POS_FRAMES, begin)
            trio = []
            for _ in range(dw.FRAMES_IN):
                ok, image = capture.read()
                if not ok:
                    break
                trio.append(image)
            if len(trio) < dw.FRAMES_IN:
                sys.exit("não consegui ler o trio que começa no frame {}".format(begin))
            position = begin + dw.FRAMES_IN
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != position:
                seek_warnings += 1
            height, width = trio[0].shape[:2]
            found = dw.detect_triplet(predict, trio, wanted, args, width, height)
            for offset in wanted:
                for x, y, conf in sorted(found[offset], key=lambda c: -c[2]):
                    pending.append((begin + offset, "{:.2f}".format(x), "{:.2f}".format(y), "{:.4f}".format(conf)))
            done += 1
            if done % args.save_every == 0:
                flush(done)
            if (done - started_done) % args.report_every == 0:
                per_job = (time.time() - started) / (done - started_done)
                print("  {}/{} passadas, {:.2f} s/passada, faltam ~{:.1f} h".format(
                    done, total, per_job, per_job * (total - done) / 3600.0))
    except KeyboardInterrupt:
        print("\ninterrompido: salvando o que já foi feito.")
    finally:
        flush(done)
        handle.close()

    elapsed = time.time() - started
    if done > started_done:
        per_job = elapsed / (done - started_done)
        print("\n{} passadas nesta execução, {:.2f} s/passada. Total do jogo: ~{:.1f} h nesta máquina.".format(
            done - started_done, per_job, per_job * total / 3600.0))
    if seek_warnings:
        print("ATENÇÃO: {} vezes o decodificador não parou na posição pedida (seek impreciso).".format(seek_warnings))
    print("feito: {}/{} passadas. Saída: {}".format(done, total, args.out))
    if done < total:
        print("Rode o mesmo comando de novo para continuar.")


if __name__ == "__main__":
    main()