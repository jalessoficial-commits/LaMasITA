"""Os dois ajustes que só a janela pública (5 minutos de GT) pode dar. Grava modelos/calibracao.json.

    python -m posse_ml.calibrar
    python -m posse_ml.calibrar --sem-transmissao

1. Peso da TV na bola parada. Os modelos foram treinados na DFL, que não tem vídeo de TV. Uma
   regressão logística na janela pública recombina o log-odds de bola parada dos modelos com duas
   frações de ±1 s (TV sem campo, TV com outra vista). São quatro números, com penalidade nos dois
   da TV: mais que isso cinco minutos não sustentam.
2. Antecipação das paradas. Com a nossa bola, a bola some quando sai, mas os jogadores só desaceleram
   depois, e as paradas decodificadas podem começar e terminar tarde. Para cada borda de parada do GT
   se mede quanto a prevista atrasou. Se quase todas atrasam (MIN_CONSENSO), as paradas passam a ser
   puxadas para trás em metade do atraso mediano, separadamente no começo e no fim; senão, ficam como
   estão.

Isto não serve para escolher entre versões do método (17 posses é pouco); só preenche o que a DFL
não tem como ensinar.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

from . import config, modelos, prever, transmissao, viterbi
from .config import D, FPS, X
from .partida import carregar_jogo

L2 = 1.0
JANELA_BORDA = 5 * FPS        # uma borda prevista casa com a do GT até 5 s de distância
MIN_CONSENSO = 0.75           # fração das bordas atrasadas para valer a correção
TETO = {"inicio": 2 * FPS, "fim": FPS}


def ajustar_transmissao(p_morta, ev, gt):
    """Regressão logística de bola morta na janela: log-odds dos modelos + as evidências `ev`."""
    avaliado = gt != X
    nomes = sorted(ev)
    cols = np.column_stack([modelos.logit(p_morta)] + [np.nan_to_num(ev[k]) for k in nomes])[avaliado]
    y = (gt[avaliado] == D).astype(np.float64)

    def perda(beta):
        z = beta[0] + cols @ beta[1:]
        return (np.sum(np.logaddexp(0.0, z) - y * z) + L2 * np.sum(beta[2:] ** 2)) / len(y)

    beta = minimize(perda, np.r_[0.0, 1.0, np.zeros(len(nomes))], method="L-BFGS-B").x
    pesos = {"intercepto": float(beta[0]), "modelo": float(beta[1])}
    pesos.update({k: float(b) for k, b in zip(nomes, beta[2:])})
    morta = y > 0
    com_tv = np.mean((beta[0] + cols @ beta[1:] > 0) == morta)
    sem_tv = np.mean((cols[:, 0] > 0) == morta)
    print("{}: acerto morta/viva na janela {:.3f} (só o modelo {:.3f})".format(
        " + ".join(nomes), com_tv, sem_tv))
    return {k: round(v, 4) for k, v in pesos.items()}


def bordas(rot, entra):
    """Frames em que uma parada começa (entra=True) ou acaba."""
    m = rot == D
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == (1 if entra else -1)) + 1


def atrasos(prev, gt):
    """{inicio: [...], fim: [...]}: previsto menos GT, em frames, para cada borda do GT na janela."""
    avaliado = np.flatnonzero(gt != X)
    a, b = avaliado[0], avaliado[-1] + 1
    out = {}
    for nome, entra in (("inicio", True), ("fim", False)):
        g = bordas(gt[a:b], entra)
        q = bordas(prev[a:b], entra)
        out[nome] = [int(q[np.argmin(np.abs(q - x))] - x) for x in g
                     if len(q) and np.min(np.abs(q - x)) <= JANELA_BORDA]
    return out


def adiantamento(lista):
    out = {}
    for nome, v in lista.items():
        v = np.array(v)
        atrasou = (v > 0).mean() if len(v) else 0.0
        if len(v) >= 4 and atrasou >= MIN_CONSENSO:
            out[nome] = int(min(TETO[nome], round(0.5 * np.median(v[v > 0]))))
        else:
            out[nome] = 0
        print("  {}: {} bordas, atrasos {} -> adiantar {} frames".format(
            nome, len(v), sorted(v.tolist()), out[nome]))
    return out


def calibrar(p, probs, plano, gt, pasta, paradas=None):
    """Ajusta os dois números na janela pública e grava calibracao.json (ou calibracao_<fonte>.json,
    com paradas de fora) em `pasta`. Devolve o que gravou."""
    from .__main__ import placar_publico
    if paradas and config.nome_paradas(paradas) == "gt":
        raise ValueError("--paradas com o próprio GT público seria ajustar o peso com a resposta")
    custos, _, _ = prever.ajustes(pasta)
    forca = prever.ler_paradas(paradas, p.n) if paradas else None
    cal = {}
    ev = prever.evidencias(plano, forca)
    if ev:
        cal["transmissao"] = ajustar_transmissao(probs[0], ev, gt)

    extra = prever.extra_da_transmissao(probs[0], plano, cal.get("transmissao"), forca)
    sem = viterbi.rotulos(viterbi.decodificar(viterbi.emissoes(*probs, extra_morta=extra),
                                              p.tempos, custos, p.kickoffs))
    print("bordas de parada na janela pública (previsto - GT, em frames):")
    cal["adiantar"] = adiantamento(atrasos(sem, gt))
    com = viterbi.adiantar_paradas(sem, p.tempos, cal["adiantar"]["inicio"], cal["adiantar"]["fim"], p.kickoffs)
    print("janela pública sem adiantar: spellF1 {:.3f}  macroF1 {:.3f}  posses {:.0f} (GT {:.0f})".format(*placar_publico(sem)))
    print("janela pública adiantando:   spellF1 {:.3f}  macroF1 {:.3f}  posses {:.0f} (GT {:.0f})".format(*placar_publico(com)))

    os.makedirs(pasta, exist_ok=True)
    destino = os.path.join(pasta, prever.arquivo_calibracao(paradas))
    with open(destino, "w", encoding="utf-8") as arq:
        json.dump(cal, arq, indent=1)
    print("gravado em", destino)
    return cal


def main(argv=None):
    from .__main__ import pm          # o __main__ põe evaluation/ no caminho do Python

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    config.argumento_classificador(ap)
    ap.add_argument("--bola", help="outro CSV da bola; padrão: a bola da instância")
    ap.add_argument("--sem-transmissao", action="store_true")
    ap.add_argument("--paradas", help="CSV de bola morta de fora (ex.: analise/out/paradas_D.csv) que entra "
                    "como evidência; grava calibracao_<fonte>.json, sem mexer no calibracao.json")
    ap.add_argument("--modelos", help="padrão: modelos/<instância>")
    args = ap.parse_args(argv)
    args.bola = args.bola or config.bola_da_instancia(args.instancia)
    args.modelos = args.modelos or config.pasta_modelos(args.instancia, args.classificador)

    if args.paradas and config.nome_paradas(args.paradas) == "gt":
        sys.exit("--paradas com o próprio GT público seria ajustar o peso com a resposta; use o modo rígido")
    gt = pm.label_codes(pm.read_label_csv(config.GT_PUBLICO)).astype(np.int8)
    p = carregar_jogo(bola=args.bola)
    probs = prever.probabilidades(p, args.modelos)
    plano = None if args.sem_transmissao else transmissao.do_jogo(p)[0]
    calibrar(p, probs, plano, gt, args.modelos, args.paradas)


if __name__ == "__main__":
    main()
