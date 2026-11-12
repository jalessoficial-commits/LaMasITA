"""Lista de paradas para rotular à mão no viewer (M5), e o leitor dos rótulos preenchidos.

    python analise/rotular_paradas.py            # gera analise/out/rotulos_manuais.csv (não sobrescreve)
    python analise/rotular_paradas.py --force    # regenera, apagando o que já foi preenchido

Como rotular, para cada linha:
  1. no viewer, "ir para frame" em `frame_ref` (ou use N / Shift+N para pular entre paradas e apitos);
  2. ache o momento em que a bola deixou de estar em jogo (apito, a bola cruzando a linha) e anote em
     `inicio_real` o frame; ache o reinício (a bola tocada na cobrança) e anote em `fim_real`;
  3. se ali NÃO houve parada (controle, apito falso), deixe inicio/fim vazios e escreva `viva` em `obs`.

As categorias misturam o que o detector já acerta e o que é duvidoso, para medir os dois:
    escanteio, lateral, tiro_de_meta         set pieces do Smart Tagging
    apito_parada                             paradas abertas por apito (freio coletivo)
    apito_sem_parada                         apitos que o detector descartou (vantagem? torcida?)
    v1_sem_evento                            paradas só pela velocidade, sem apito nem bola na linha
    controle_viva                            trechos longe de qualquer evento (devem ser jogo)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FPS = 25
OUT = os.path.join(HERE, "out", "rotulos_manuais.csv")
COTAS = {"escanteio": 8, "lateral": 8, "tiro_de_meta": 4, "apito_parada": 6, "apito_sem_parada": 6,
         "v1_sem_evento": 6, "controle_viva": 6}


def mmss(frame):
    s = frame / FPS
    return "{:02d}:{:05.2f}".format(int(s // 60), s % 60)


def gerar(S, tabela, seed=0):
    """tabela = saída de paradas.detectar_paradas. Fora da janela do GT público (ela já tem gabarito)."""
    import rotulos as R
    import eventos as E
    rng = np.random.default_rng(seed)
    gt = R.gt_public()
    cov = np.flatnonzero(gt != "X")
    fora_gt = lambda f: not (cov[0] - 60 * FPS <= f <= cov[-1] + 60 * FPS)
    rows = []

    def add(cat, frame, ini=None, fim=None, info=""):
        rows.append(dict(categoria=cat, frame_ref=int(frame), tempo_video=mmss(frame),
                         previsto_inicio=ini, previsto_fim=fim, info=info))

    def previsto(f):
        k = tabela[(tabela.inicio <= f + 30 * FPS) & (tabela.fim >= f - 5 * FPS)]
        return (int(k.inicio.iloc[0]), int(k.fim.iloc[0])) if len(k) else (None, None)

    sp = R.set_pieces()
    for cat, tipo in (("escanteio", "Corner Kick"), ("lateral", "Throw-in"), ("tiro_de_meta", "Goal Kick")):
        fr = [f for f in sp.start_frame[sp.type == tipo] if fora_gt(f)]
        for f in rng.choice(fr, min(COTAS[cat], len(fr)), replace=False):
            add(cat, f - 20 * FPS, *previsto(f), info="start_frame do Smart Tagging = {}".format(f))

    ap = tabela[(tabela.fonte_inicio == "apito") & tabela.inicio.map(fora_gt)]
    for _, r in ap.sample(min(COTAS["apito_parada"], len(ap)), random_state=seed).iterrows():
        add("apito_parada", r.inicio - 3 * FPS, int(r.inicio), int(r.fim), info="fim por {}".format(r.fonte_fim))

    abertos = set(ap.inicio.values) | set(tabela[tabela.fonte_inicio == "apito"].inicio.values)
    descart = [e for e in E.eventos_apito(S) if fora_gt(e["frame"])
               and not any(abs(e["frame"] - a) <= 2 * FPS for a in abertos)]
    for i in rng.choice(len(descart), min(COTAS["apito_sem_parada"], len(descart)), replace=False):
        e = descart[i]
        add("apito_sem_parada", e["frame"] - 3 * FPS, info="apito SNR {:.0f} dB descartado".format(e["forca"]))

    v1 = tabela[(tabela.tipo == "sem_evento") & tabela.inicio.map(fora_gt)]
    for _, r in v1.sample(min(COTAS["v1_sem_evento"], len(v1)), random_state=seed).iterrows():
        add("v1_sem_evento", r.inicio - 5 * FPS, int(r.inicio), int(r.fim))

    dead = np.zeros(S.N, bool)
    for r in tabela.itertuples():
        dead[max(0, r.inicio - 30 * FPS):r.fim + 30 * FPS] = True
    livres = np.flatnonzero(S.graded & ~dead)
    livres = [f for f in livres[::FPS * 10] if fora_gt(f)]
    for f in rng.choice(livres, min(COTAS["controle_viva"], len(livres)), replace=False):
        add("controle_viva", f, info="nenhum evento a ±30 s")

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)   # embaralha: rotular às cegas
    df.insert(0, "id", np.arange(1, len(df) + 1))
    for c in ("previsto_inicio", "previsto_fim"):
        df[c] = df[c].astype("Int64")
    df["inicio_real"] = ""
    df["fim_real"] = ""
    df["obs"] = ""
    return df


def carregar(path=OUT):
    """Linhas rotuladas: inicio_real/fim_real numéricos, ou obs == 'viva'."""
    df = pd.read_csv(path, dtype={"obs": str}).fillna({"obs": ""})
    df["inicio_real"] = pd.to_numeric(df.inicio_real, errors="coerce")
    df["fim_real"] = pd.to_numeric(df.fim_real, errors="coerce")
    feito = df.inicio_real.notna() & df.fim_real.notna() | df.obs.str.lower().str.contains("viva")
    return df[feito].copy()


OUT_POSSE = os.path.join(HERE, "out", "rotulos_posse.csv")


OUT_BLOCOS = os.path.join(HERE, "out", "rotulos_blocos.csv")
OUT_CLIPES = os.path.join(HERE, "out", "rotulos_clipes.csv")


def carregar_posse(path=OUT_POSSE):
    """Marcas de posse do viewer (teclas H / A / D): frame, time ('H' | 'A' | 'D'), obs; em ordem.
    Cada marca vale do seu frame até a próxima."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["frame", "time", "obs"])
    df = pd.read_csv(path, dtype={"time": str, "obs": str}).fillna({"obs": ""})
    df = df[df.time.isin(["H", "A", "D"])]
    return df.astype({"frame": int}).sort_values("frame").reset_index(drop=True)


def carregar_blocos(path=OUT_BLOCOS):
    """Blocos de posse rotulados por completo no viewer: bloco, inicio, fim (frames), obs."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["bloco", "inicio", "fim", "obs"])
    df = pd.read_csv(path, dtype={"obs": str}).fillna({"obs": ""})
    return df.astype({"inicio": int, "fim": int}).sort_values("inicio").reset_index(drop=True)


def carregar_clipes(path=OUT_CLIPES):
    """Vereditos do painel "Clipes D" do viewer: inicio, fim (o clipe previsto), fonte (a predição),
    veredito ('certo' | 'falso' | 'ajustado'), inicio_real, fim_real, obs. None se ainda não existe."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"fonte": str, "veredito": str, "obs": str}).fillna({"obs": "", "veredito": ""})
    for c in ("inicio_real", "fim_real"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df.veredito != ""].astype({"inicio": int, "fim": int}).sort_values("inicio").reset_index(drop=True)


def gt_blocos(n_frames, posse=None, blocos=None):
    """Gabarito dos blocos no formato oficial: um rótulo H/A/D por frame dentro dos blocos e 'X' fora.
    Dentro de um bloco cada frame tem o rótulo da última marca até ele; antes da primeira marca, 'X'.
    Serve para pontuar com evaluation/possession_metrics como se fosse o GT (gt=gt_blocos(N))."""
    posse = carregar_posse() if posse is None else posse
    blocos = carregar_blocos() if blocos is None else blocos
    out = np.full(n_frames, "X", dtype="<U1")
    for b in blocos.itertuples():
        m = posse[(posse.frame >= b.inicio) & (posse.frame <= b.fim)]
        for k, r in enumerate(m.itertuples()):
            fim = m.frame.iloc[k + 1] if k + 1 < len(m) else b.fim + 1
            out[r.frame:min(fim, n_frames)] = r.time
    return out


def avaliar(df, dead):
    """Por linha rotulada: erro de início e fim do trecho previsto que mais cobre a parada real, e
    para 'viva' a fração do trecho de ±10 s marcada D (deveria ser 0)."""
    import arvore as T
    pred = T.runs(dead)
    out = []
    for r in df.itertuples():
        if np.isnan(r.inicio_real):
            f = int(r.frame_ref)
            out.append(dict(id=r.id, categoria=r.categoria, real="viva", falso_D_frac=float(dead[f - 250:f + 250].mean())))
            continue
        a, b = int(r.inicio_real), int(r.fim_real)
        best = max(((min(b, e) - max(a, s), s, e) for s, e in pred if min(b, e) > max(a, s)), default=None)
        if best is None:
            out.append(dict(id=r.id, categoria=r.categoria, real="parada", cobertura=0.0))
        else:
            _, s, e = best
            out.append(dict(id=r.id, categoria=r.categoria, real="parada", cobertura=float(dead[a:b].mean()),
                            erro_inicio_s=(s - a) / FPS, erro_fim_s=(e - b) / FPS))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if os.path.exists(OUT) and not args.force:
        n = len(carregar())
        raise SystemExit("{} já existe ({} linhas rotuladas). Use --force para regenerar.".format(OUT, n))
    import eventos as E
    import paradas as P
    from features_coletivas import load_all
    X, m = load_all()
    S = E.Sinais(X, m["index"])
    _, tab = P.detectar_paradas(S, P.Config())
    df = gerar(S, tab)
    df.to_csv(OUT, index=False)
    print("{} paradas para rotular -> {}".format(len(df), OUT))
    print(df.categoria.value_counts().to_string())


if __name__ == "__main__":
    main()
