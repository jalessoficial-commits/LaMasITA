"""Treina e compara os classificadores de uma instância, de uma vez.

    python -m posse_ml.comparar_classificadores                      # todos, GPU quando der
    python -m posse_ml.comparar_classificadores --classificadores xgboost gru tcn
    python -m posse_ml.comparar_classificadores --so-comparar        # não treina; compara o que já existe
    python -m posse_ml.comparar_classificadores --retreinar xgboost  # treina de novo este (sem nome: todos)

Para cada classificador:
  1. treina (python -m posse_ml.treinar, num processo à parte, com a validação rápida; --completo
     para a validação inteira), a não ser que já esteja treinado. Na GPU quando o classificador e a
     máquina deixam: xgboost, catboost e as redes; o hgb e o lightgbm ficam na CPU.
  2. calibra na janela pública, sem e com o detector de paradas;
  3. grava as três submissões (bola morta própria, detector rígido, detector suave) em
     evaluation/results/ e mede cada uma na janela pública.
No fim, uma tabela com a validação na DFL e a janela pública, gravada também em
modelos/<instância>/comparacao_classificadores.md. O jogo é lido e os atributos calculados uma vez só.

Se um treino falhar (pacote faltando, memória), os outros seguem; o log de cada treino fica em
cache/logs/.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from . import atributos, calibrar, config, modelos, prever, transmissao
from .config import A, D, H, X
from .partida import carregar_jogo

ORDEM = ("hgb", "xgboost", "lightgbm", "gru", "tcn", "catboost", "lstm", "transformer")
PACOTE = {"hgb": "sklearn", "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
          "gru": "torch", "lstm": "torch", "tcn": "torch", "transformer": "torch"}
# --teste (só o ponta a ponta usa): modelos pequenos, para conferir que tudo encaixa
PEQUENOS = {"hgb": "max_iter=40,min_samples_leaf=50", "xgboost": "n_estimators=40",
            "lightgbm": "n_estimators=40,min_child_samples=50", "catboost": "iterations=40",
            "gru": "epocas=2", "lstm": "epocas=2", "tcn": "epocas=2", "transformer": "epocas=2"}


def gpu(classificador):
    """True se dá para treinar este classificador na GPU desta máquina."""
    try:
        if classificador in ("gru", "lstm", "tcn", "transformer"):
            import torch
            return bool(torch.cuda.is_available())
        if classificador == "catboost":
            from catboost.utils import get_gpu_device_count
            return get_gpu_device_count() > 0
        if classificador == "xgboost":
            import xgboost
            if not xgboost.build_info().get("USE_CUDA") or shutil.which("nvidia-smi") is None:
                return False
            return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except Exception:
        return False
    return False


def treinado(pasta):
    return os.path.isfile(os.path.join(pasta, "decodificador.json")) and all(
        os.path.isfile(os.path.join(pasta, "conjuntos", c, "manifesto.json")) for c in ("direto", "invertido"))


def gravado_desde(pasta, t0):
    """O treino terminou de gravar depois de t0: decodificador e os dois conjuntos."""
    arquivos = [os.path.join(pasta, "decodificador.json")] + [
        os.path.join(pasta, "conjuntos", c, "manifesto.json") for c in ("direto", "invertido")]
    return all(os.path.isfile(a) and os.path.getmtime(a) >= t0 - 1 for a in arquivos)


def rodar(modulo_e_args, log):
    """python -m <...> com a saída na tela e no log. Devolve o código de saída."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with open(log, "w", encoding="utf-8") as arq:
        proc = subprocess.Popen([sys.executable, "-m"] + modulo_e_args, cwd=config.RAIZ, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for linha in proc.stdout:
            sys.stdout.write(linha)
            sys.stdout.flush()
            arq.write(linha)
        return proc.wait()


def treinar_pasta(inst, clf, pasta, hiper="", completo=False, sem_gpu=False, passo=None, nome=None, extra=None):
    """Roda o treinar.py de `clf` com os modelos em `pasta` (validação rápida, a não ser com completo;
    na GPU quando dá). Devolve (deu certo, descrição do tempo)."""
    usa_gpu = not sem_gpu and gpu(clf)
    nome = nome or os.path.basename(os.path.normpath(pasta))
    print("\n=== {}: treinando na {}{}".format(nome, "GPU" if usa_gpu else "CPU",
                                             " ({})".format(hiper) if hiper else ""), flush=True)
    cmd = ["posse_ml.treinar", "--instancia", inst, "--classificador", clf, "--modelos", pasta]
    cmd += ["--hiper", hiper] if hiper else []
    cmd += [] if completo else ["--rapido"]
    cmd += ["--gpu"] if usa_gpu else []
    cmd += ["--passo", str(passo)] if passo else []
    cmd += list(extra or [])
    t0 = time.time()
    codigo = rodar(cmd, os.path.join(config.CACHE, "logs", "treinar_{}_{}.log".format(inst, nome)))
    if codigo and gravado_desde(pasta, t0):
        # no Windows, o PyTorch com CUDA às vezes derruba o processo na saída (0xC0000409), depois
        # de tudo gravado: o treino valeu
        print("=== {}: o processo caiu ao sair (código {}), mas os modelos foram gravados".format(nome, codigo))
        codigo = 0
    if codigo:
        print("=== {}: o treino falhou (código {})".format(nome, codigo))
        return False, "falhou"
    return True, "{:.0f} min ({})".format((time.time() - t0) / 60, "GPU" if usa_gpu else "CPU")


def _num(v):
    return "{:.3f}".format(v).replace(".", ",")


def _placar(pl):
    return "{} / {} ({:.0f})".format(_num(pl[0]), _num(pl[1]), pl[2]) if pl else "—"


NOTA = ("Janela pública: spellF1 / macroF1 (posses previstas; GT 17). Quem decide é a validação na "
        "DFL (7 jogos inteiros); a janela tem 17 posses e serve só para conferir.")


def tabela(linhas):
    """A tabela em markdown: validação na DFL e as três versões na janela pública."""
    cab = ["classificador", "treino", "validação DFL (spellF1 / macroF1)", "janela, bola morta própria",
           "janela, detector rígido", "janela, detector suave"]
    tab = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    for ln in linhas:
        if "erro" in ln:
            tab.append("| {} | {} | erro: {} | | | |".format(ln["clf"], ln.get("treino", ""), ln["erro"][:60]))
            continue
        v = ln["validacao"]
        val = "{} / {}{}".format(_num(v[0]), _num(v[1]), "" if ln["rapida"] else " (completa)") if v[0] else "—"
        tab.append("| {} | {} | {} | {} | {} | {} |".format(
            ln["clf"], ln.get("treino", ""), val, _placar(ln.get("propria")), _placar(ln.get("rigido")),
            _placar(ln.get("suave"))))
    return "\n".join(tab)


def jogo_e_atributos(inst):
    """O jogo, os atributos, o plano da TV e o GT público, lidos uma vez para todos os classificadores."""
    from .__main__ import pm
    print("\n=== lendo o jogo e calculando os atributos (uma vez para todos)", flush=True)
    p = carregar_jogo(bola=config.bola_da_instancia(inst))
    x, nomes = atributos.matriz(p)
    plano, desl = transmissao.do_jogo(p)
    print("transmissão: tático = transmissão + {}".format(" / ".join(str(d) for d in desl)))
    gt = pm.label_codes(pm.read_label_csv(config.GT_PUBLICO)).astype(np.int8)
    return p, x, nomes, plano, gt


def avaliar(p, x, nomes, plano, gt, inst, clf, paradas, teto=False):
    """Calibra, grava as submissões (própria, e rígida e suave se houver detector) e mede na janela.
    teto=True mede também com a bola morta do GT (rígido), sem gravar arquivo: só para comparar o H/A."""
    from .__main__ import placar_publico, pm
    pasta = config.pasta_modelos(inst, clf)
    print("\n=== {}: calibrando e gravando as submissões".format(clf), flush=True)
    probs = modelos.probabilidades(x, nomes, pasta, p.tempos)
    with open(os.path.join(pasta, "decodificador.json"), encoding="utf-8") as arq:
        dec = json.load(arq)
    linha = {"clf": clf, "validacao": (dec.get("spellF1_validacao"), dec.get("macroF1_validacao")),
             "rapida": dec.get("rapido")}
    # o modelo sozinho, antes do Viterbi, frame a frame na janela: bola morta/viva e H/A com a bola viva
    avaliado = gt != X
    viva = (gt == H) | (gt == A)
    linha["frames_morta"] = float(np.mean((probs[0][avaliado] > 0.5) == (gt[avaliado] == D)))
    linha["frames_ha"] = float(np.mean((probs[1][viva] > 0.5) == (gt[viva] == H)))
    print("  por frame na janela: morta/viva {:.3f}, H/A com a bola viva {:.3f}".format(
        linha["frames_morta"], linha["frames_ha"]))
    versoes = [("propria", None, None)]
    if paradas:
        versoes += [("rigido", paradas, "rigido"), ("suave", paradas, "suave")]
    calibrar.calibrar(p, probs, plano, gt, pasta)
    for nome, par, modo in versoes:
        if modo == "suave":
            calibrar.calibrar(p, probs, plano, gt, pasta, par)
        rot = prever.rotular(p, probs, plano, pasta, paradas=par, modo=modo)
        saida = prever.gravar(config.nome_saida(inst, clf, par, modo), rot)
        problema = pm.validate_submission(pm.read_label_csv(saida), len(rot))
        linha[nome] = None if problema else placar_publico(rot)
        print("  {:7s} {}  {}".format(nome, os.path.basename(saida),
                                      "INVÁLIDA: " + problema if problema else _placar(linha[nome])))
    if teto:
        linha["teto"] = placar_publico(prever.rotular(p, probs, plano, pasta, paradas=config.GT_PUBLICO,
                                                      modo="rigido"))
        print("  teto    (bola morta do GT, não gravado)  {}".format(_placar(linha["teto"])))
    return linha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config.argumento_instancia(ap)
    ap.add_argument("--classificadores", nargs="+", choices=ORDEM, default=list(ORDEM))
    ap.add_argument("--retreinar", nargs="*", choices=ORDEM,
                    help="treina de novo mesmo que já exista (sem nome: todos os da lista)")
    ap.add_argument("--so-comparar", action="store_true", help="não treina nada")
    ap.add_argument("--completo", action="store_true", help="validação inteira, sem --rapido")
    ap.add_argument("--sem-gpu", action="store_true")
    ap.add_argument("--teste", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--paradas", default=config.PARADAS_EQUIPE,
                    help="detector de paradas para as versões rígida e suave (padrão: analise/out/paradas_D.csv)")
    args = ap.parse_args(argv)
    inst = args.instancia
    paradas = args.paradas if args.paradas and os.path.isfile(args.paradas) else None
    if paradas is None:
        print("AVISO: sem o detector de paradas ({}); só a versão com bola morta própria".format(args.paradas))
    if not os.path.isfile(config.GT_PUBLICO):
        sys.exit("preciso do GT público em {} para calibrar e comparar".format(config.GT_PUBLICO))

    # 1. treino
    tempo_treino = {}
    for clf in args.classificadores:
        if importlib.util.find_spec(PACOTE[clf]) is None:
            print("\n=== {}: pacote {} não instalado, pulando (requirements-classificadores.txt)".format(
                clf, PACOTE[clf]))
            continue
        pasta = config.pasta_modelos(inst, clf)
        refazer = args.retreinar is not None and (not args.retreinar or clf in args.retreinar)
        if args.so_comparar or (treinado(pasta) and not refazer):
            print("\n=== {}: já treinado em {}".format(clf, pasta) if treinado(pasta) else
                  "\n=== {}: não treinado, pulando".format(clf))
            continue
        _, tempo_treino[clf] = treinar_pasta(inst, clf, pasta, PEQUENOS[clf] if args.teste else "",
                                             args.completo, args.sem_gpu, 3 if args.teste else None, clf)

    # 2 e 3. calibração, submissões e placar na janela, com o jogo lido uma vez
    prontos = [c for c in args.classificadores if treinado(config.pasta_modelos(inst, c))
               and importlib.util.find_spec(PACOTE[c]) is not None]
    if not prontos:
        sys.exit("nenhum classificador treinado para comparar")
    p, x, nomes, plano, gt = jogo_e_atributos(inst)

    linhas = []
    for clf in prontos:
        try:
            linha = avaliar(p, x, nomes, plano, gt, inst, clf, paradas)
            linha["treino"] = tempo_treino.get(clf, "já treinado")
            linhas.append(linha)
        except Exception as erro:          # um classificador com problema não derruba a comparação
            print("=== {}: falhou na comparação: {}".format(clf, erro))
            linhas.append({"clf": clf, "treino": tempo_treino.get(clf, "?"), "erro": str(erro)})

    texto = tabela(linhas)
    print("\n" + texto + "\n\n" + NOTA)
    validos = [ln for ln in linhas if "erro" not in ln and ln["validacao"][0]]
    if validos:
        m = max(validos, key=lambda ln: ln["validacao"][0])
        print("melhor na validação da DFL: {} (spellF1 {})".format(m["clf"], _num(m["validacao"][0])))
    destino = os.path.join(config.pasta_modelos(inst), "comparacao_classificadores.md")
    with open(destino, "w", encoding="utf-8", newline="\n") as arq:
        arq.write("# Classificadores, instância {}\n\n{}\n\n{}\n".format(inst, texto, NOTA))
    print("tabela gravada em", destino)
    falhas = [c for c, t in tempo_treino.items() if t == "falhou"] + [ln["clf"] for ln in linhas if "erro" in ln]
    if falhas:
        sys.exit("com problema: " + ", ".join(sorted(set(falhas))))


if __name__ == "__main__":
    main()
