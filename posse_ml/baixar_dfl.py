"""Baixa os sete jogos abertos da DFL (Sportec Open Data, CC-BY 4.0) para posse_ml/dfl/.

    python -m posse_ml.baixar_dfl

Bassek, Rein, Weber e Memmert, "An integrated dataset of spatiotemporal and event data in elite
soccer", Scientific Data 12:195 (2025), doi:10.6084/m9.figshare.28196177. Só os arquivos de
posições e de informações do jogo, cerca de 2,5 GB. Arquivo que já está lá com o MD5 certo é pulado.
Os dados não vão para o Git (posse_ml/.gitignore).

Mostra o progresso a cada 15 s. Conexão parada por 60 s cai e é retomada de onde parou (o
arquivo .parcial fica na pasta); interromper e rodar de novo também continua de onde parou.
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait

from . import config

API = "https://api.figshare.com/v2/articles/28196177/files?page_size=1000"
QUERO = ("positions_raw_observed", "matchinformation")


def _md5(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as arq:
        for bloco in iter(lambda: arq.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


TIMEOUT = 60          # s sem receber nada: a conexão caiu, tenta de novo
TENTATIVAS = 8
_lock = threading.Lock()
_recebido = [0]


def _conta(n):
    with _lock:
        _recebido[0] += n


def _uma_vez(url, parcial, tamanho):
    """Baixa para `parcial`, continuando de onde parou se o servidor aceitar Range."""
    ja = os.path.getsize(parcial) if os.path.isfile(parcial) else 0
    cab = {"User-Agent": "posse_ml/1.0"}
    if ja:
        cab["Range"] = "bytes={}-".format(ja)
    req = urllib.request.Request(url, headers=cab)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        continua = ja and resp.status == 206
        if ja and not continua:
            _conta(-ja)              # o servidor recomeçou do zero
        with open(parcial, "ab" if continua else "wb") as out:
            while True:
                bloco = resp.read(1 << 20)
                if not bloco:
                    break
                out.write(bloco)
                _conta(len(bloco))
    # a conexão pode fechar no meio sem erro nenhum: só o tamanho mostra
    if tamanho and os.path.getsize(parcial) < tamanho:
        raise OSError("veio {} de {} bytes".format(os.path.getsize(parcial), tamanho))


def _baixar(item, pasta):
    destino = os.path.join(pasta, item["name"])
    tamanho = item.get("size", 0)
    if os.path.isfile(destino) and _md5(destino) == item.get("computed_md5"):
        _conta(tamanho)
        return "ok (já estava)  " + item["name"]
    parcial = destino + ".parcial"
    if os.path.isfile(parcial):
        _conta(os.path.getsize(parcial))
    url = item.get("download_url") or "https://ndownloader.figshare.com/files/{}".format(item["id"])
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            _uma_vez(url, parcial, tamanho)
            break
        except (OSError, http.client.HTTPException) as erro:
            print("  {}: conexão caiu ({}), tentativa {}/{}".format(item["name"], erro, tentativa, TENTATIVAS),
                  flush=True)
            time.sleep(min(5 * tentativa, 30))
    else:
        return "ERRO, desisti     " + item["name"]
    if item.get("computed_md5") and _md5(parcial) != item["computed_md5"]:
        _conta(-os.path.getsize(parcial))
        os.remove(parcial)
        return "ERRO de MD5 (apagado, rode de novo) " + item["name"]
    os.replace(parcial, destino)
    return "baixado {:>5} MB {}".format(tamanho // 1_000_000, item["name"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pasta", default=config.DFL)
    ap.add_argument("--conexoes", type=int, default=6, help="downloads em paralelo (cada um é lento)")
    args = ap.parse_args(argv)
    os.makedirs(args.pasta, exist_ok=True)
    req = urllib.request.Request(API, headers={"User-Agent": "posse_ml/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        itens = [i for i in json.load(resp) if any(q in i["name"] for q in QUERO)]
    if not itens:
        sys.exit("a lista de arquivos do figshare veio sem posições; confira {}".format(API))
    itens.sort(key=lambda i: -i.get("size", 0))
    total = sum(i.get("size", 0) for i in itens)
    print("{} arquivos, {:.0f} MB no total".format(len(itens), total / 1e6), flush=True)
    t0, antes = time.time(), None
    with ThreadPoolExecutor(max_workers=args.conexoes) as pool:
        pendentes = {pool.submit(_baixar, i, args.pasta) for i in itens}
        while pendentes:
            prontos, pendentes = wait(pendentes, timeout=15)
            for f in prontos:
                print(f.result(), flush=True)
            agora = _recebido[0]
            ritmo = (agora - antes) / 15 / 1e6 if antes is not None and not prontos else None
            print("  {:.0f} de {:.0f} MB ({:.0%}){}".format(
                agora / 1e6, total / 1e6, agora / max(total, 1),
                ", {:.1f} MB/s".format(ritmo) if ritmo is not None else ""), flush=True)
            antes = agora
    print("{} arquivos em {} ({:.0f} s)".format(len(itens), args.pasta, time.time() - t0))


if __name__ == "__main__":
    main()
