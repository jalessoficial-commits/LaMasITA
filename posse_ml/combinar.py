"""Combina classificadores já treinados (média dos log-odds) e valida a combinação na DFL.

    python -m posse_ml.combinar                                   # gru + tcn
    python -m posse_ml.combinar --classificadores gru tcn xgboost
    python -m posse_ml.combinar --so-validacao                    # só a tabela da DFL

Não treina nada. A validação usa as probabilidades que o treino de cada classificador deixou no
cache (cache/validacao/), jogo a jogo: a combinação de cada jogo de fora é a média, em log-odds, das
previsões dos membros para aquele jogo, que nunca viram esse jogo no treino. Os custos do Viterbi da
combinação são escolhidos de novo nessa validação, com a mesma grade e a mesma regra do platô do
treinar.py.

A combinação fica em modelos/<instância>/<a+b>/ (membros.json, decodificador.json e as calibrações)
e funciona como um classificador comum: python -m posse_ml --modelos modelos/viterbi/gru+tcn. Depois
da validação, calibra, grava as três submissões (posse_ml_<instância>_<a+b>_...) e mede na janela,
junto com os membros, para a tabela sair completa.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

from . import comparar_classificadores as C, config, modelos, perfil as P, treinar as T, viterbi


def _json(caminho):
    with open(caminho, encoding="utf-8") as arq:
        return json.load(arq)


def validacao_do_cache(inst, clf):
    """(jogos, semente, perfil, custos, {jogo: (p_morta, p_mandante, p_reinicio)}) da validação
    que o treino de `clf` deixou no cache."""
    pasta = config.pasta_modelos(inst, clf)
    caminho = os.path.join(pasta, "decodificador.json")
    if not os.path.isfile(caminho):
        sys.exit("{} não está treinado ({} não existe)".format(clf, caminho))
    dec = _json(caminho)
    caminho_perfil = os.path.join(config.pasta_modelos(inst), "perfil_bola.json")
    perf = P.carregar(caminho_perfil) if os.path.isfile(caminho_perfil) else P.PADRAO
    jogos, semente = dec["jogos"], dec.get("semente", 0)
    chave = dec.get("chave_validacao")
    if not chave:
        # treinos de antes desta versão não gravavam a chave: refaz a conta com o que o treino usou
        man = _json(os.path.join(pasta, "conjuntos", "direto", "manifesto.json"))
        tipo = dec.get("classificador") or man.get("classificador") or clf
        hv, passo_v = T.config_validacao(man.get("passo", T.PASSO), man.get("hiper") or {}, tipo, dec.get("rapido"),
                                         semente)
        chave = T.chave_validacao(perf, passo_v, hv, jogos, semente, tipo, dec.get("rapido"))
    probs, faltam = {}, []
    for j in jogos:
        arq = T.arquivo_validacao(j, chave)
        if not os.path.isfile(arq):
            faltam.append(j)
            continue
        z = config.ler_npz(arq)
        probs[j] = (z["p_morta"], z["p_mandante"], z["p_reinicio"])
    if faltam:
        sys.exit("a validação de {} não está no cache para {}. Refaça só ela com:\n  python -m "
                 "posse_ml.treinar --so-validacao{} --instancia {} --classificador {}".format(
                     clf, ", ".join(faltam), " --rapido" if dec.get("rapido") else "", inst, clf))
    return jogos, semente, perf, dec, probs


def gabaritos(jogos, semente, perf):
    """{jogo: (rótulos, tempos)} da cópia 0 de cada jogo, a mesma que a validação decodificou."""
    out = {}
    for i, j in enumerate(jogos):
        caminho = T.preparar((j, i, 100000 * semente, perf, config.DFL))[0]
        with np.load(caminho) as z:
            out[j] = (z["rot"], [tuple(int(v) for v in t) for t in z["tempos"]])
    return out


def placares(probs, gab, grade):
    """(jogos, custos, 4): spellF1, macroF1, posses previstas e reais, para cada custo da grade."""
    jogos = list(probs)
    tab = np.zeros((len(jogos), len(grade), 4))
    for i, j in enumerate(jogos):
        rot, tempos = gab[j]
        estados = viterbi.decodificar(viterbi.emissoes(*probs[j]), tempos, grade)
        for g in range(len(grade)):
            tab[i, g] = T.placar(rot, viterbi.rotulos(estados[g]))
    return tab


def combinar(inst, membros, nome):
    """Valida na DFL a média dos `membros` (pastas em modelos/<inst>/: "gru", "tcn_s1"...), escolhe os
    custos do Viterbi, grava modelos/<inst>/<nome>/ e imprime a tabela jogo a jogo. Devolve
    {versão: (spellF1 por jogo, macroF1 médio, posses previstas/reais, custos)}."""
    val = {c: validacao_do_cache(inst, c) for c in membros}
    jogos, semente, perf = val[membros[0]][:3]
    for c in membros[1:]:
        if val[c][0] != jogos or val[c][1] != semente or T._chave_perfil(val[c][2]) != T._chave_perfil(perf):
            sys.exit("{} e {} foram validados em jogos, semente ou perfil diferentes; treine de novo um "
                     "deles".format(membros[0], c))
    print("lendo os rótulos dos {} jogos da DFL (cópias do cache)".format(len(jogos)), flush=True)
    gab = gabaritos(jogos, semente, perf)

    por_jogo = {}
    for c in membros:
        dec = val[c][3]
        tab = placares(val[c][4], gab, [dec["custos"]])[:, 0]
        por_jogo[c] = (tab[:, 0], tab[:, 1].mean(), (tab[:, 2] / tab[:, 3]).mean(), dec["custos"])
    print("decodificando {} com a grade de custos".format(nome), flush=True)
    comb = {j: tuple(modelos.media_logit([val[c][4][j][k] for c in membros]) for k in range(3)) for j in jogos}
    grade = [dict(zip(T.GRADE, v)) for v in itertools.product(*T.GRADE.values())]
    tab = placares(comb, gab, grade)
    e = T.escolher(tab[:, :, 0].mean(axis=0), grade)
    por_jogo[nome] = (tab[:, e, 0], tab[:, e, 1].mean(), (tab[:, e, 2] / tab[:, e, 3]).mean(), grade[e])

    versoes = membros + [nome]
    larg = max(len(v) for v in versoes + ["jogo"]) + 2
    print("\nvalidação na DFL, cada jogo de fora (spellF1, com os custos de cada versão)")
    print("jogo".ljust(10) + "".join(v.rjust(larg) for v in versoes))
    for i, j in enumerate(jogos):
        print(j.ljust(10) + "".join("{:.3f}".format(por_jogo[v][0][i]).rjust(larg) for v in versoes))
    print("média".ljust(10) + "".join("{:.3f}".format(por_jogo[v][0].mean()).rjust(larg) for v in versoes))
    print("macroF1".ljust(10) + "".join("{:.3f}".format(por_jogo[v][1]).rjust(larg) for v in versoes))
    print("posses".ljust(10) + "".join("{:.2f}".format(por_jogo[v][2]).rjust(larg) for v in versoes))
    melhor_membro = np.max([por_jogo[c][0] for c in membros], axis=0)
    print("\ncustos de {}: {}".format(nome, grade[e]))
    print("{} passa o melhor membro de cada jogo em {} de {}; contra a média de cada membro: {}".format(
        nome, int(np.sum(por_jogo[nome][0] > melhor_membro)), len(jogos),
        ", ".join("{} ({:+.3f})".format(c, por_jogo[nome][0].mean() - por_jogo[c][0].mean()) for c in membros)))

    pasta = config.pasta_modelos(inst, nome)
    os.makedirs(pasta, exist_ok=True)
    with open(os.path.join(pasta, "membros.json"), "w", encoding="utf-8") as arq:
        json.dump({"classificadores": membros,
                   "pastas": [os.path.relpath(config.pasta_modelos(inst, c), pasta) for c in membros]}, arq, indent=1)
    with open(os.path.join(pasta, "decodificador.json"), "w", encoding="utf-8") as arq:
        json.dump({"custos": grade[e], "spellF1_validacao": round(float(tab[:, e, 0].mean()), 4),
                   "macroF1_validacao": round(float(tab[:, e, 1].mean()), 4), "jogos": jogos,
                   "rapido": all(val[c][3].get("rapido") for c in membros), "classificador": nome,
                   "membros": membros, "semente": semente}, arq, indent=1)
    print("gravado em", pasta)
    return por_jogo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificadores", nargs="+", default=["gru", "tcn"],
                    help="pastas em modelos/<instância>/ (gru, tcn, xgboost, tcn_s1...)")
    ap.add_argument("--so-validacao", action="store_true", help="não calibra nem grava as submissões")
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst, membros = args.instancia, list(dict.fromkeys(args.classificadores))
    if len(membros) < 2:
        sys.exit("uma combinação precisa de pelo menos dois classificadores")
    nome = "+".join(membros)
    combinar(inst, membros, nome)
    if args.so_validacao:
        return

    # calibração, submissões e janela, para os membros e a combinação
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("sem o GT público ({}), parei antes de calibrar".format(config.GT_PUBLICO))
    p, x, nomes, plano, gt = C.jogo_e_atributos(inst)
    linhas = [C.avaliar(p, x, nomes, plano, gt, inst, v, paradas) for v in membros + [nome]]
    print("\n" + C.tabela(linhas) + "\n\n" + C.NOTA)


if __name__ == "__main__":
    main()
