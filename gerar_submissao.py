"""Gera a submissão da equipe a partir dos dados do desafio, do zero até <time>.csv.

    python gerar_submissao.py                    # tudo, na ordem; o que já foi feito é pulado
    python gerar_submissao.py --refazer          # refaz todas as etapas (a detecção da bola retoma de onde parou)
    python gerar_submissao.py --time LaMasITA    # nome do arquivo final (padrão: LaMasITA)

Pré-requisitos (README.md, seção "Submissão da equipe"): os dados em data/ (ou $MTAH_DATA_DIR), as
dependências de requirements-submissao.txt e os pesos do detector de bola em
tracking_bola/pesos/football-ball-detection.pt.

Etapas (cada uma é um script do repositório, rodado com este mesmo Python):

 1. tracking_bola/run_champion_full.py     detector de bola do Roboflow (YOLOv8) em todos os frames
                                           avaliados, máscara do campo e fusão com o ATD (horas; na GPU, se houver)
 2. tracking_bola/export_ball.py           a fusão no formato do ATD (data/tactical/..._ball_tracker.csv)
 3. tracking_bola/track_temporal.py        rastro da bola por Viterbi -> tracking_bola/out/viterbi_bola.csv
 4. viewer/preprocess.py, viewer/features.py, analise/features_coletivas.py, analise/audio_apito.py
                                           sinais do detector de bola morta (ATD, homografia, apito)
 5. analise/avaliar_v3.py --exportar       detector de bola morta: D v2 (entrada do posse_ml) e D v5
 6. python -m posse_ml                     H/A: TCN (média de seis sementes, treinada só na DFL),
                                           modelo/campeao_tcn, bola morta do D v2 em modo rígido
 7. analise/casar_campeao.py               D v5 + H/A da TCN, posses < 1 s engolidas
 8. <time>.csv na raiz, conferido com evaluation/validate_submission.py

Nenhuma etapa lê o GT para decidir rótulos: o GT público só aparece nos placares impressos.
"""

import argparse
import os
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
GAME = "BAR-ATH"


def caminho(*partes):
    return os.path.join(RAIZ, *partes)


def rodar(descricao, args, saidas, refazer):
    """Roda `args` (lista; o primeiro item é um script ou '-m') se alguma saída faltar."""
    if not refazer and all(os.path.exists(s) for s in saidas):
        print("== {} (já feito, pulando)".format(descricao), flush=True)
        return
    print("\n== {}\n   {}".format(descricao, " ".join([os.path.basename(PY)] + args)), flush=True)
    subprocess.run([PY] + args, cwd=RAIZ, check=True)
    faltam = [s for s in saidas if not os.path.exists(s)]
    if faltam:
        sys.exit("a etapa '{}' terminou sem gravar: {}".format(descricao, ", ".join(faltam)))


def conferir_entradas():
    sys.path.insert(0, RAIZ)
    from games import require
    require(["atd_raw"])                       # sai com a mensagem do games.py se data/ estiver vazio
    pesos = caminho("tracking_bola", "pesos", "football-ball-detection.pt")
    if not os.path.isfile(pesos):
        sys.exit("faltam os pesos do detector de bola em {} (README.md, seção 'Submissão da equipe')".format(pesos))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--time", default="LaMasITA", help="nome da equipe = nome do CSV final")
    ap.add_argument("--refazer", action="store_true", help="refaz as etapas mesmo com as saídas presentes")
    args = ap.parse_args()
    r = args.refazer
    conferir_entradas()

    tb = caminho("tracking_bola", "out")
    rodar("1. bola: detector do Roboflow + máscara + fusão", ["tracking_bola/run_champion_full.py"],
          [caminho("tracking_bola", "out", "full_rf_campo.csv"), caminho("tracking_bola", "out", "full_bola.csv")], r)
    from games import require
    pasta_atd = os.path.dirname(require(["atd_raw"])[0])
    rodar("2. bola: exportação da fusão", ["tracking_bola/export_ball.py"],
          [os.path.join(pasta_atd, "{}_tactical_ball_tracker.csv".format(GAME))], r)
    rodar("3. bola: rastro por Viterbi", ["tracking_bola/track_temporal.py", "--method", "viterbi"],
          [os.path.join(tb, "viterbi_bola.csv")], r)

    vd = caminho("viewer", "data")
    rodar("4a. ATD e homografia em binário", ["viewer/preprocess.py"], [os.path.join(vd, "atd_tactical.bin")], r)
    rodar("4b. features por frame", ["viewer/features.py"], [os.path.join(vd, "features_tactical.bin")], r)
    rodar("4c. features coletivas", ["analise/features_coletivas.py"],
          [os.path.join(vd, "features_coletivas_tactical.bin")], r)
    rodar("4d. apito no áudio", ["analise/audio_apito.py"], [os.path.join(vd, "apito_tactical.json")], r)

    trk = caminho("analise", "out", "tracking")
    d_v2, d_v5 = os.path.join(trk, "D_v2.csv"), os.path.join(trk, "D_v5.csv")
    rodar("5. bola morta (v2 e v5)", ["analise/avaliar_v3.py", "--exportar", "--sem-cache"],
          [d_v2, d_v5, os.path.join(trk, "goleiro_v5.csv")], r)

    res = caminho("evaluation", "results")
    camp = os.path.join(res, "campeao_tcn.csv")
    rodar("6. posse H/A: TCN, bola morta v2 rígida",
          ["-m", "posse_ml", "--instancia", "viterbi", "--modelos", "modelo/campeao_tcn",
           "--paradas", os.path.relpath(d_v2, RAIZ), "--modo-paradas", "rigido", "--out", os.path.relpath(camp, RAIZ)],
          [camp], True)           # sempre: evaluation/results/campeao_tcn.csv está no Git, então nunca falta

    casado = os.path.join(res, "casado_tcn_v5.csv")
    rodar("7. D v5 + H/A da TCN", ["analise/casar_campeao.py", "--campeoes", "campeao_tcn", "--d", "v5",
                                    "--min-posse-s", "1", "--exportar"], [casado], True)

    final = caminho("{}.csv".format(args.time))
    shutil.copyfile(casado, final)
    print("\n== 8. validação de", os.path.basename(final), flush=True)
    subprocess.run([PY, "evaluation/validate_submission.py", final], cwd=RAIZ, check=True)
    print("\nsubmissão pronta: {}".format(final))


if __name__ == "__main__":
    main()
