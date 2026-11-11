"""Baixa os 3 jogos abertos da Metrica Sports (github.com/metrica-sports/sample-data) para
posse_ml/metrica_dados (~190 MB).

    python -m posse_ml.baixar_metrica

Rastreamento a 25 Hz e eventos anotados de 3 jogos anônimos. O repositório pede uso responsável e
citação da fonte. Os rótulos de posse são derivados dos eventos em metrica.py.
"""

import argparse
import os
import sys
import time

from . import config
from .baixar_dfl import TENTATIVAS, _uma_vez

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
ARQUIVOS = {
    "Sample_Game_1": ["Sample_Game_1_RawEventsData.csv", "Sample_Game_1_RawTrackingData_Home_Team.csv",
                      "Sample_Game_1_RawTrackingData_Away_Team.csv"],
    "Sample_Game_2": ["Sample_Game_2_RawEventsData.csv", "Sample_Game_2_RawTrackingData_Home_Team.csv",
                      "Sample_Game_2_RawTrackingData_Away_Team.csv"],
    "Sample_Game_3": ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"],
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pasta", default=config.METRICA)
    args = ap.parse_args(argv)
    falhas = 0
    for jogo, arquivos in ARQUIVOS.items():
        os.makedirs(os.path.join(args.pasta, jogo), exist_ok=True)
        for nome in arquivos:
            destino = os.path.join(args.pasta, jogo, nome)
            if os.path.isfile(destino) and os.path.getsize(destino) > 1000:
                print("ok (já estava)  " + nome)
                continue
            t0 = time.time()
            for tentativa in range(1, TENTATIVAS + 1):
                try:
                    _uma_vez(BASE + jogo + "/" + nome, destino + ".parcial", 0)
                    os.replace(destino + ".parcial", destino)
                    print("baixado {:>4} MB {} ({:.0f} s)".format(os.path.getsize(destino) // 1_000_000, nome,
                                                                 time.time() - t0), flush=True)
                    break
                except OSError as erro:
                    print("  {}: conexão caiu ({}), tentativa {}/{}".format(nome, erro, tentativa, TENTATIVAS),
                          flush=True)
                    time.sleep(min(5 * tentativa, 30))
            else:
                print("ERRO, desisti   " + nome)
                falhas += 1
    if falhas:
        sys.exit("{} arquivo(s) não baixaram; rode de novo".format(falhas))
    print("pronto em {}. Confira os rótulos com: python -m posse_ml.metrica".format(args.pasta))


if __name__ == "__main__":
    main()
