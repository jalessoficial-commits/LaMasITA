"""Treina os classificadores na DFL degradada e escolhe os custos do Viterbi.

    python -m posse_ml.baixar_dfl                 # uma vez, ~2,5 GB
    python -m posse_ml.medir_bola                 # perfil do nosso tracker (modelos/perfil_bola.json)
    python -m posse_ml.treinar                    # validação deixando um jogo de fora + modelos finais
    python -m posse_ml.treinar --rapido           # validação com menos árvores (bem mais rápida)
    python -m posse_ml.treinar --so-validacao     # só a tabela da validação
    python -m posse_ml.treinar --pular-validacao  # só os modelos finais, com os custos já salvos

Cópias: cada jogo da DFL é degradado três vezes com sorteios diferentes, duas na orientação original
e uma com os lados trocados. O conjunto "direto" usa as duas primeiras; o "invertido" usa a primeira
e a trocada, o que o impede de aprender que o mandante tem mais a bola. A submissão usa os dois.

Validação: para cada jogo, os modelos são treinados nos outros e o jogo de fora é decodificado com
toda a grade de custos. O custo escolhido é o de troca mais cara entre os que ficam a TOLERANCIA do
melhor spellF1 médio. Motivo: os rótulos da DFL trocam de posse com muito mais frequência que o
gabarito do desafio, então o custo que a DFL prefere tende a gerar posses demais aqui; dentro do
empate, vale ficar no lado conservador. Nada aqui lê os dados do desafio.
"""

import argparse
import gc
import hashlib
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import atributos, config, degradacao, dfl, metrica, modelos, perfil as P, viterbi
from .config import X

sys.path.insert(0, os.path.join(config.RAIZ, "evaluation"))
import possession_metrics as pm  # noqa: E402

COPIAS = {0: False, 1: False, 2: True}          # cópia -> invertida?
CONJUNTOS = {"direto": (0, 1), "invertido": (0, 2)}
PASSO = 2                                        # um frame a cada PASSO no treino
GRADE = {"troca": [2.0, 4.0, 6.0, 9.0, 13.0], "parada": [2.0, 4.0, 8.0, 12.0], "reinicio": [1.0, 3.0]}
TOLERANCIA = 0.005
ARVORES_RAPIDO = 150
PASSO_RAPIDO = 4                                 # na validação rápida, um frame a cada 4


def _chave_perfil(perf):
    """Perfil da bola + versão da degradação: as duas coisas que definem uma cópia de treino."""
    return hashlib.sha1(json.dumps([perf, degradacao.VERSAO], sort_keys=True).encode()).hexdigest()[:8]


def preparar(args):
    """As cópias degradadas de um jogo, com atributos, no cache. Devolve {cópia: caminho do .npz}.

    Uma tarefa por jogo, não por cópia: assim o XML de cada jogo é lido uma vez só, e dois
    processos nunca leem nem gravam o mesmo arquivo do cache ao mesmo tempo.
    """
    jogo, i, semente_base, perf, pasta_dfl = args
    out, p = {}, None
    for copia, invertida in COPIAS.items():
        semente = semente_base + 1000 * i + copia
        destino = os.path.join(config.CACHE, "treino", "{}_c{}_s{}_{}.npz".format(
            jogo, copia, semente, _chave_perfil(perf)))
        out[copia] = destino
        if os.path.isfile(destino):
            continue
        t0 = time.time()
        if p is None:
            p = metrica.carregar(jogo) if metrica.eh_metrica(jogo) else dfl.carregar(jogo, pasta_dfl)
        base = dfl.inverter(p) if invertida else p
        q, par = degradacao.degradar(base, np.random.default_rng(semente), perf)
        x, nomes = atributos.matriz(q)
        config.gravar_npz(destino, x=x, rot=base.rotulos, prox=modelos.proximo_dono(base.rotulos),
                          tempos=np.array(base.tempos), nomes=np.array(nomes), par=json.dumps(par))
        print("  {} cópia {}/{} pronta ({:.0f} s)".format(jogo, copia + 1, len(COPIAS), time.time() - t0),
              flush=True)
    return out


def _ler(caminho):
    z = config.ler_npz(caminho)
    return {"x": z["x"], "rot": z["rot"], "prox": z["prox"],
            "tempos": [tuple(int(v) for v in t) for t in z["tempos"]], "nomes": list(z["nomes"])}


def _linhas(caminhos, passo):
    """As linhas de treino das cópias, já no array final (sem juntar pedaços, que dobraria a
    memória): primeiro conta as linhas de cada cópia, depois preenche."""
    indices = []
    for c in caminhos:
        with np.load(c) as z:
            indices.append(np.flatnonzero(z["rot"] != X)[::passo])
    total = sum(len(i) for i in indices)
    x = r = p = None
    ini = 0
    for c, i in zip(caminhos, indices):
        d = _ler(c)
        if x is None:
            x = np.empty((total, d["x"].shape[1]), dtype=np.float32)
            r = np.empty(total, dtype=d["rot"].dtype)
            p = np.empty(total, dtype=d["prox"].dtype)
        fim = ini + len(i)
        x[ini:fim], r[ini:fim], p[ini:fim] = d["x"][i], d["rot"][i], d["prox"][i]
        ini = fim
        del d
    return x, r, p


def treinar_conjuntos(arquivos, jogos, passo, hiper, semente=0, classificador="hgb", dispositivo="cpu",
                      rapido=False):
    out = {}
    for nome, copias in CONJUNTOS.items():
        caminhos = [arquivos[(j, c)] for j in jogos for c in copias]
        t0 = time.time()
        if classificador in modelos.SEQUENCIAIS:
            from . import sequencia
            out[nome] = sequencia.treinar(caminhos, classificador, semente, hiper, dispositivo, rapido)
            print("  conjunto {}: {} cópias, {:.0f} s".format(nome, len(caminhos), time.time() - t0), flush=True)
            continue
        x, r, pr = _linhas(caminhos, passo)
        out[nome] = modelos.treinar(x, r, pr, semente, hiper, classificador, dispositivo)
        print("  conjunto {}: {} linhas, {:.0f} s".format(nome, len(r), time.time() - t0), flush=True)
        del x, r, pr
        gc.collect()
    return out


def _valor(texto):
    """'600' -> 600, '0.1' -> 0.1, 'true' -> True, o resto fica texto."""
    for tipo in (int, float):
        try:
            return tipo(texto)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(texto.lower(), texto)


def placar(gt, rot):
    r = pm.evaluate(gt.astype(np.int8), rot.astype(np.int8))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def config_validacao(passo, hiper, classificador, rapido, semente=0):
    """(hiperparâmetros, passo) da validação: com --rapido, menos árvores e menos frames.

    Nas redes, o que não muda a validação sai da chave do cache: com epocas_rapido dado, as épocas
    do modelo final (a validação treina epocas_rapido); e os valores iguais ao padrão (semente igual
    à do treino, epocas_rapido=5). Assim, variantes que só mudam as épocas finais dividem a mesma
    validação."""
    hv = dict(hiper)
    # árvores: a validação rápida usa ARVORES_RAPIDO, a não ser que o número de árvores tenha sido
    # dado no --hiper (aí a validação usa o mesmo número do modelo final)
    if rapido and classificador in modelos.ITERACOES and modelos.ITERACOES[classificador] not in hv:
        hv[modelos.ITERACOES[classificador]] = ARVORES_RAPIDO
    if classificador in modelos.SEQUENCIAIS:
        if rapido and "epocas_rapido" in hv:
            hv.pop("epocas", None)
        if hv.get("epocas_rapido") == 5:
            hv.pop("epocas_rapido")
        if hv.get("semente") == semente:
            hv.pop("semente")
    return hv, (max(passo, PASSO_RAPIDO) if rapido else passo)


def chave_validacao(perf, passo_v, hv, jogos, semente, classificador, rapido):
    """A chave das probabilidades da validação no cache (cache/validacao/<jogo>_<chave>.npz)."""
    return hashlib.sha1(json.dumps([_chave_perfil(perf), passo_v, hv, jogos, semente, classificador, rapido],
                                   sort_keys=True).encode()).hexdigest()[:8]


def arquivo_validacao(jogo, chave):
    return os.path.join(config.CACHE, "validacao", "{}_{}.npz".format(jogo, chave))


def escolher(media, grade):
    """O índice do custo escolhido pela regra do platô."""
    melhor = float(np.max(media))
    perto = [i for i in range(len(grade)) if media[i] >= melhor - TOLERANCIA]
    return max(perto, key=lambda i: (grade[i]["troca"], grade[i]["parada"], -grade[i]["reinicio"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--so-validacao", action="store_true")
    ap.add_argument("--pular-validacao", action="store_true",
                    help="treina os modelos finais com os custos do decodificador.json (ou os padrão)")
    ap.add_argument("--rapido", action="store_true", help="validação com {} árvores (ou menos épocas, nas "
                    "redes) e um frame a cada {}; os modelos finais não mudam".format(ARVORES_RAPIDO, PASSO_RAPIDO))
    ap.add_argument("--jogos", nargs="*", help="IDs dos jogos (padrão: os de --dados)")
    ap.add_argument("--dados", choices=("dfl", "metrica", "dfl+metrica"), default="dfl",
                    help="jogos de treino: os da DFL (padrão), os 3 da Metrica (baixar_metrica.py) ou todos")
    config.argumento_instancia(ap)
    ap.add_argument("--perfil", help="padrão: modelos/<instância>/perfil_bola.json")
    ap.add_argument("--passo", type=int, default=PASSO)
    ap.add_argument("--processos", type=int, default=1,
                    help="jogos preparados em paralelo; cada um usa ~1,5 GB de RAM no pico")
    ap.add_argument("--hiper", default="", help="ajustes no classificador, ex.: max_iter=600 (hgb), "
                    "n_estimators=600 (xgboost), epocas=20 (redes)")
    config.argumento_classificador(ap)
    ap.add_argument("--gpu", action="store_true", help="treina na GPU (xgboost, catboost e as redes); a "
                    "previsão continua na CPU")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--pasta-dfl", default=config.DFL)
    ap.add_argument("--modelos", help="padrão: modelos/<instância>")
    args = ap.parse_args(argv)
    args.modelos = args.modelos or config.pasta_modelos(args.instancia, args.classificador)
    args.perfil = args.perfil or os.path.join(config.pasta_modelos(args.instancia), "perfil_bola.json")
    dispositivo = "cuda" if args.gpu else "cpu"
    print("instância {}, classificador {}: modelos em {}".format(args.instancia, args.classificador, args.modelos))

    hiper = {}
    for item in filter(None, args.hiper.split(",")):
        k, v = item.split("=")
        hiper[k.strip()] = _valor(v.strip())
    if os.path.isfile(args.perfil):
        perf = P.carregar(args.perfil)
        print("perfil da bola:", perf.get("descricao", args.perfil))
    else:
        perf = P.PADRAO
        print("AVISO: {} não existe; uso o perfil padrão, que não foi medido. Rode antes "
              "python -m posse_ml.medir_bola --instancia {}".format(args.perfil, args.instancia))

    # os da DFL sempre primeiro e na mesma ordem: a semente de cada cópia depende da posição do jogo
    jogos = args.jogos or ((dfl.jogos(args.pasta_dfl) if "dfl" in args.dados else []) +
                           (metrica.jogos() if "metrica" in args.dados else []))
    if "metrica" in args.dados and not metrica.jogos():
        sys.exit("não achei os jogos da Metrica em {} (python -m posse_ml.baixar_metrica)".format(config.METRICA))
    if len(jogos) < 2:
        sys.exit("preciso de pelo menos 2 jogos de treino (python -m posse_ml.baixar_dfl ou baixar_metrica)")
    print("jogos de treino ({}): {}".format(len(jogos), " ".join(jogos)))
    tarefas = [(j, i, 100000 * args.semente, perf, args.pasta_dfl) for i, j in enumerate(jogos)]
    t0 = time.time()
    print("preparando as cópias degradadas ({} jogos x {}); as que já estão no cache são puladas".format(
        len(jogos), len(COPIAS)), flush=True)
    if args.processos > 1:
        with ProcessPoolExecutor(max_workers=min(args.processos, len(tarefas))) as pool:
            por_jogo = list(pool.map(preparar, tarefas))
    else:
        por_jogo = [preparar(t) for t in tarefas]
    arquivos = {(j, c): caminho for j, d in zip(jogos, por_jogo) for c, caminho in d.items()}
    nomes = _ler(arquivos[(jogos[0], 0)])["nomes"]
    print("{} jogos x {} cópias prontos em {:.0f} s ({} atributos)".format(
        len(jogos), len(COPIAS), time.time() - t0, len(nomes)), flush=True)

    grade = [dict(zip(GRADE, v)) for v in itertools.product(*GRADE.values())]
    custos = None
    if not args.pular_validacao:
        # cada rodada fica no cache com uma chave da configuração: se o treino cair no meio,
        # rodar de novo pula as rodadas já feitas
        hv, passo_v = config_validacao(args.passo, hiper, args.classificador, args.rapido, args.semente)
        chave = chave_validacao(perf, passo_v, hv, jogos, args.semente, args.classificador, args.rapido)
        tabela = np.zeros((len(jogos), len(grade), 4))
        for i, fora in enumerate(jogos):
            print("validação {}/{}: {} de fora".format(i + 1, len(jogos), fora), flush=True)
            teste = _ler(arquivos[(fora, 0)])
            feito = arquivo_validacao(fora, chave)
            if os.path.isfile(feito):
                z = config.ler_npz(feito)
                probs = (z["p_morta"], z["p_mandante"], z["p_reinicio"])
                print("  (já feita, do cache)")
            else:
                trios = treinar_conjuntos(arquivos, [j for j in jogos if j != fora], passo_v, hv, args.semente,
                                          args.classificador, dispositivo, args.rapido)
                saidas = [modelos.prever(t, teste["x"], teste["tempos"]) for t in trios.values()]
                probs = tuple(modelos.media_logit([s[k] for s in saidas]) for k in range(3))
                del trios, saidas
                gc.collect()
                config.gravar_npz(feito, p_morta=probs[0], p_mandante=probs[1], p_reinicio=probs[2])
            estados = viterbi.decodificar(viterbi.emissoes(*probs), teste["tempos"], grade)
            for g in range(len(grade)):
                tabela[i, g] = placar(teste["rot"], viterbi.rotulos(estados[g]))
            b = int(np.argmax(tabela[i, :, 0]))
            print("  melhor spellF1 {:.3f}  macroF1 {:.3f}  posses {:.0f}/{:.0f} com {}".format(
                *tabela[i, b], grade[b]), flush=True)
        media = tabela[:, :, 0].mean(axis=0)
        print("\nmédia nos jogos de fora (spellF1, macroF1, posses previstas / reais)")
        for g in np.argsort(-media)[:10]:
            print("  {:.3f}  {:.3f}  {:.2f}  {}".format(media[g], tabela[:, g, 1].mean(),
                                                      (tabela[:, g, 2] / tabela[:, g, 3]).mean(), grade[g]))
        e = escolher(media, grade)
        custos = grade[e]
        print("\nescolhido (platô de {:.3f}, troca mais cara): {}  spellF1 {:.3f}".format(TOLERANCIA, custos, media[e]))
        for i, j in enumerate(jogos):
            print("  {}  spellF1 {:.3f}  macroF1 {:.3f}  posses {:.0f} / {:.0f}".format(j, *tabela[i, e]))
        if not args.so_validacao:
            os.makedirs(args.modelos, exist_ok=True)
            with open(os.path.join(args.modelos, "decodificador.json"), "w", encoding="utf-8") as arq:
                json.dump({"custos": custos, "spellF1_validacao": round(float(media[e]), 4),
                           "macroF1_validacao": round(float(tabela[:, e, 1].mean()), 4),
                           "jogos": jogos, "rapido": bool(args.rapido), "classificador": args.classificador,
                           "semente": args.semente, "chave_validacao": chave},
                          arq, indent=1)
    if args.so_validacao:
        return

    print("modelos finais, nos {} jogos".format(len(jogos)), flush=True)
    trios = treinar_conjuntos(arquivos, jogos, args.passo, hiper, args.semente, args.classificador, dispositivo)
    for nome, trio in trios.items():
        modelos.salvar(trio, os.path.join(args.modelos, "conjuntos", nome), nomes,
                       {"jogos": jogos, "perfil": perf.get("descricao", ""), "passo": args.passo,
                        "copias": list(CONJUNTOS[nome]), "hiper": hiper}, args.classificador)
    print("salvo em {}".format(os.path.join(args.modelos, "conjuntos")))


if __name__ == "__main__":
    main()
