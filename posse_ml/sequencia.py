"""Redes de sequência (GRU, LSTM, TCN e transformer) sobre os mesmos 100 atributos por frame dos modelos de árvore.

As árvores olham um frame por vez, e o contexto entra só pelos atributos de janela (médias de 2 s
antes e depois, "desde" e "até"). Aqui a rede lê o trecho inteiro e aprende sozinha o que importa
antes e depois de cada frame. É a ideia dos trabalhos que usam GRU sobre atributos agregados por
time: como os atributos já são por time, a rede não precisa de trilhas de jogador, que o ATD não tem.

Uma rede só responde às três perguntas (morta, time, reinicio): uma saída para cada, com a perda de
cada uma só nos frames em que ela faz sentido.

  gru   GRU bidirecional de 2 camadas
  lstm  LSTM bidirecional de 2 camadas (a mesma ideia da GRU, com uma memória a mais)
  tcn   rede convolucional temporal não causal: blocos residuais com dilatação 1, 2, 4, ..., 32,
        que alcançam ~10 s para cada lado
  transformer  encoder pequeno (3 camadas, 4 cabeças) com atenção sobre a janela inteira (~20 s),
        viés de posição relativa aprendido (cada cabeça aprende o quanto pesa a distância entre dois
        frames) e uma convolução na entrada para dar o prior local que transformer puro não tem;
        normalização antes de cada subcamada (pre-LN), que treina melhor com poucos dados

Detalhes:
  - um frame a cada PASSO (12,5 Hz): a posse não muda em 40 ms, e a sequência fica com metade do
    tamanho. Na previsão, a rede roda nas duas fases (frames pares e ímpares) e as saídas se
    intercalam, então todo frame tem a sua probabilidade.
  - entrada: atributos padronizados (média e desvio do treino), NaN vira 0, mais uma coluna 0/1 de
    "faltando" para cada atributo que às vezes falta.
  - treino em janelas de 256 passos (hiper "janela") sorteadas das cópias degradadas; previsão em
    janelas do mesmo tamanho, com metade de sobreposição, ficando com o miolo de cada uma.
  - a previsão roda na CPU, mesmo que o treino tenha sido na GPU: a submissão sai igual em qualquer
    máquina.
"""

import json
import os
import time

import numpy as np

from .config import A, D, H

PASSO = 2
JANELA = 256
HIPER = {"epocas": 10, "epocas_rapido": 5, "largura": 64, "lote": 32, "lr": 2e-3, "dropout": 0.1,
         "janela": JANELA}
# o que muda por tipo de rede (o --hiper ainda passa por cima)
# blocos: blocos da TCN, com dilatação 1, 2, 4, ...; cada um soma 4 x dilatação passos ao alcance
# (6 blocos: ~±10 s; 4: ~±2,5 s; 7: ~±20 s, que pede janela de 512 passos). camadas: das recorrentes
# e do transformer. janela: passos (a 12,5 Hz) de cada trecho de treino e de previsão.
HIPER_TIPO = {"transformer": {"lr": 1e-3, "camadas": 3, "cabecas": 4}, "tcn": {"blocos": 6},
              "gru": {"camadas": 2}, "lstm": {"camadas": 2}}


def hiper_de(tipo, hiper=None):
    return {**HIPER, **HIPER_TIPO.get(tipo, {}), **(hiper or {})}


def _torch():
    try:
        import torch
    except ImportError:
        raise SystemExit("as redes (gru, lstm, tcn, transformer) precisam do PyTorch: python -m pip install torch "
                         "(veja posse_ml/requirements-classificadores.txt)")
    return torch


def _rede(tipo, d_in, h):
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional
    RECORRENTES = {"gru": nn.GRU, "lstm": nn.LSTM}
    largura, dropout, janela = int(h["largura"]), float(h["dropout"]), int(h["janela"])

    class BlocoTCN(nn.Module):
        def __init__(self, c, dilatacao):
            super().__init__()
            self.a = nn.Conv1d(c, c, 3, padding=dilatacao, dilation=dilatacao)
            self.b = nn.Conv1d(c, c, 3, padding=dilatacao, dilation=dilatacao)
            self.sai = nn.Dropout(dropout)

        def forward(self, x):                      # (B, C, T)
            y = self.sai(torch.relu(self.a(x)))
            return torch.relu(x + self.b(y))

    class Atencao(nn.Module):
        """Atenção multi-cabeça com viés de posição relativa: um número aprendido por cabeça para cada
        distância entre dois frames (até ±janela), somado ao placar da atenção."""
        def __init__(self, d, cabecas):
            super().__init__()
            self.cabecas = cabecas
            self.qkv = nn.Linear(d, 3 * d)
            self.sai = nn.Linear(d, d)
            self.vies = nn.Parameter(torch.zeros(cabecas, 2 * janela - 1))

        def forward(self, x):                      # (B, T, d)
            b, t, d = x.shape
            q, k, v = self.qkv(x).view(b, t, 3, self.cabecas, d // self.cabecas).permute(2, 0, 3, 1, 4)
            i = torch.arange(t, device=x.device)
            rel = (i[None, :] - i[:, None]).clamp(-(janela - 1), janela - 1) + janela - 1
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.vies[:, rel][None],
                                               dropout_p=dropout if self.training else 0.0)
            return self.sai(y.transpose(1, 2).reshape(b, t, d))

    class BlocoTransformer(nn.Module):
        def __init__(self, d, cabecas):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.atencao = Atencao(d, cabecas)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
            self.sai = nn.Dropout(dropout)

        def forward(self, x):                      # pre-LN
            x = x + self.sai(self.atencao(self.n1(x)))
            return x + self.sai(self.ff(self.n2(x)))

    class CorpoTransformer(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.local = nn.Conv1d(d, d, 5, padding=2)
            self.blocos = nn.Sequential(*[BlocoTransformer(d, int(h["cabecas"])) for _ in range(int(h["camadas"]))])
            self.fim = nn.LayerNorm(d)

        def forward(self, x):                      # (B, T, d)
            x = x + torch.relu(self.local(x.transpose(1, 2))).transpose(1, 2)
            return self.fim(self.blocos(x))

    class Rede(nn.Module):
        def __init__(self):
            super().__init__()
            self.entrada = nn.Linear(d_in, largura)
            if tipo in RECORRENTES:
                camadas = int(h["camadas"])
                self.corpo = RECORRENTES[tipo](largura, largura, num_layers=camadas, batch_first=True,
                                               bidirectional=True, dropout=dropout if camadas > 1 else 0.0)
                saida = 2 * largura
            elif tipo == "transformer":
                self.corpo = CorpoTransformer(largura)
                saida = largura
            else:
                self.corpo = nn.Sequential(*[BlocoTCN(largura, 2 ** i) for i in range(int(h["blocos"]))])
                saida = largura
            self.saida = nn.Linear(saida, 3)

        def forward(self, x):                      # (B, T, d_in) -> (B, T, 3) logits
            h = torch.relu(self.entrada(x))
            if tipo in RECORRENTES:
                h, _ = self.corpo(h)
            elif tipo == "transformer":
                h = self.corpo(h)
            else:
                h = self.corpo(h.transpose(1, 2)).transpose(1, 2)
            return self.saida(h)

    return Rede()


class Normalizador:
    def __init__(self, media, desvio, com_falta):
        self.media = np.asarray(media, dtype=np.float32)
        self.desvio = np.asarray(desvio, dtype=np.float32)
        self.com_falta = np.asarray(com_falta, dtype=np.int64)

    @classmethod
    def ajustar(cls, xs, rng, amostra=400000):
        """Média, desvio e quais colunas às vezes faltam, numa amostra das linhas de treino."""
        linhas = []
        por_copia = max(1, amostra // len(xs))
        for x in xs:
            i = rng.choice(len(x), size=min(por_copia, len(x)), replace=False)
            linhas.append(x[i])
        a = np.concatenate(linhas).astype(np.float64)
        with np.errstate(invalid="ignore"):
            media = np.nan_to_num(np.nanmean(a, axis=0))
            desvio = np.nan_to_num(np.nanstd(a, axis=0), nan=1.0)
        desvio = np.where(desvio > 1e-6, desvio, 1.0)
        falta = np.isnan(a).mean(axis=0)
        return cls(media, desvio, np.flatnonzero(falta > 0.001))

    @property
    def d_entrada(self):
        return len(self.media) + len(self.com_falta)

    def __call__(self, x):
        z = (x - self.media) / self.desvio
        falta = np.isnan(x[:, self.com_falta]).astype(np.float32)
        z = np.clip(np.nan_to_num(z, nan=0.0), -10.0, 10.0)
        return np.concatenate([z, falta], axis=1).astype(np.float32)


def _alvos(rot, prox):
    """(3, n) alvos 0/1 e (3, n) máscaras de onde cada pergunta vale."""
    viva = (rot == H) | (rot == A)
    morta = rot == D
    alvo = np.stack([morta, rot == H, prox == 1]).astype(np.float32)
    mascara = np.stack([viva | morta, viva, morta & (prox >= 0)]).astype(np.float32)
    return alvo, mascara


class ModeloSequencia:
    def __init__(self, tipo, rede, norm, hiper):
        self.tipo, self.rede, self.norm, self.hiper = tipo, rede, norm, hiper
        self.janela = int(hiper_de(tipo, hiper)["janela"])

    def _logits(self, z):
        """Logits (T, 3) de uma sequência inteira, em janelas de self.janela passos com metade de
        sobreposição. De cada janela fica o miolo (sem o primeiro e o último quarto), que é onde a rede vê contexto
        dos dois lados; nas pontas da sequência fica a ponta também."""
        torch = _torch()
        J = self.janela
        t = len(z)
        if t <= J:
            return self.rede(torch.from_numpy(z[None])).numpy()[0]
        inicios = list(range(0, t - J + 1, J // 2))
        if inicios[-1] + J < t:
            inicios.append(t - J)
        soma = np.zeros((t, 3))
        peso = np.zeros(t)
        meio = J // 4
        for k in range(0, len(inicios), 64):
            grupo = inicios[k:k + 64]
            lote = np.stack([z[s:s + J] for s in grupo])
            lg = self.rede(torch.from_numpy(lote)).numpy()
            for j, s in enumerate(grupo):
                lo = s + (0 if s == 0 else meio)
                hi = s + J - (0 if s + J == t else meio)
                soma[lo:hi] += lg[j, lo - s:hi - s]
                peso[lo:hi] += 1
        return soma / np.maximum(peso, 1)[:, None]

    def prever_partida(self, x, tempos):
        """(p_morta, p_mandante, p_reinicio) por frame; fora dos tempos, 0,5."""
        torch = _torch()
        self.rede.eval()
        self.rede.to("cpu")
        out = np.full((3, len(x)), 0.5, dtype=np.float64)
        with torch.no_grad():
            for _, a, b in tempos:
                for fase in range(PASSO):
                    idx = np.arange(a + fase, b + 1, PASSO)
                    if len(idx):
                        out[:, idx] = 1.0 / (1.0 + np.exp(-self._logits(self.norm(x[idx])).T))
        return out[0], out[1], out[2]

    def salvar(self, pasta):
        torch = _torch()
        os.makedirs(pasta, exist_ok=True)
        torch.save(self.rede.state_dict(), os.path.join(pasta, "rede.pt"))
        np.savez(os.path.join(pasta, "normalizacao.npz"), media=self.norm.media, desvio=self.norm.desvio,
                 com_falta=self.norm.com_falta)
        with open(os.path.join(pasta, "rede.json"), "w", encoding="utf-8") as arq:
            json.dump({"tipo": self.tipo, "hiper": self.hiper, "passo": PASSO, "janela": self.janela}, arq, indent=1)


def carregar(pasta):
    torch = _torch()
    with open(os.path.join(pasta, "rede.json"), encoding="utf-8") as arq:
        cfg = json.load(arq)
    with np.load(os.path.join(pasta, "normalizacao.npz")) as z:
        norm = Normalizador(z["media"], z["desvio"], z["com_falta"])
    rede = _rede(cfg["tipo"], norm.d_entrada, hiper_de(cfg["tipo"], cfg["hiper"]))
    rede.load_state_dict(torch.load(os.path.join(pasta, "rede.pt"), map_location="cpu", weights_only=True))
    rede.eval()
    return ModeloSequencia(cfg["tipo"], rede, norm, cfg["hiper"])


def _sequencias(caminhos, norm):
    """Para cada tempo de cada cópia: (entrada (T, d) float16, alvos (3, T), máscaras (3, T)), um frame a
    cada PASSO."""
    from .config import ler_npz
    out = []
    for c in caminhos:
        d = ler_npz(c)
        alvo, mascara = _alvos(d["rot"], d["prox"])
        for t in d["tempos"]:
            _, a, b = (int(v) for v in t)
            idx = np.arange(a, b + 1, PASSO)
            if len(idx) >= 8:
                out.append((norm(d["x"][idx]).astype(np.float16), alvo[:, idx], mascara[:, idx]))
        del d
    return out


def treinar(caminhos, tipo, semente=0, hiper=None, dispositivo="cpu", rapido=False):
    """Treina uma rede nas cópias degradadas (caminhos dos .npz do cache de treino)."""
    torch = _torch()
    from .config import ler_npz
    h = hiper_de(tipo, hiper)
    # validação rápida: epocas_rapido, a não ser que só epocas tenha sido dado (aí vale nas duas)
    dado = hiper or {}
    if rapido and ("epocas_rapido" in dado or "epocas" not in dado):
        epocas = int(h["epocas_rapido"])
    else:
        epocas = int(h["epocas"])
    if dispositivo == "cuda" and not torch.cuda.is_available():
        print("  (sem GPU disponível para o PyTorch; treinando na CPU)")
        dispositivo = "cpu"
    # --hiper semente=N troca só o sorteio da rede (pesos iniciais, janelas, dropout); o --semente do
    # treinar.py também muda as cópias degradadas
    semente = int(h.get("semente", semente))
    rng = np.random.default_rng(semente)
    torch.manual_seed(semente)

    norm = Normalizador.ajustar([ler_npz(c)["x"] for c in caminhos[:4]], rng)
    seqs = _sequencias(caminhos, norm)
    tamanhos = np.array([len(s[0]) for s in seqs], dtype=np.float64)
    J = int(h["janela"])
    passos_por_epoca = max(1, int(tamanhos.sum() // J // h["lote"]))

    rede = _rede(tipo, norm.d_entrada, h).to(dispositivo)
    otim = torch.optim.AdamW(rede.parameters(), lr=float(h["lr"]), weight_decay=1e-4)
    total = epocas * passos_por_epoca
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, max_lr=float(h["lr"]), total_steps=total)
    perda_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    t0 = time.time()
    for ep in range(epocas):
        rede.train()
        soma = 0.0
        for _ in range(passos_por_epoca):
            lote = rng.choice(len(seqs), size=int(h["lote"]), p=tamanhos / tamanhos.sum())
            xs, ys, ms = [], [], []
            for i in lote:
                z, alvo, mascara = seqs[i]
                s = int(rng.integers(0, max(len(z) - J, 0) + 1))
                e = min(s + J, len(z))
                pad = J - (e - s)
                xs.append(np.pad(z[s:e].astype(np.float32), ((0, pad), (0, 0))))
                ys.append(np.pad(alvo[:, s:e], ((0, 0), (0, pad))))
                ms.append(np.pad(mascara[:, s:e], ((0, 0), (0, pad))))
            x = torch.from_numpy(np.stack(xs)).to(dispositivo)
            y = torch.from_numpy(np.stack(ys)).to(dispositivo).transpose(1, 2)
            m = torch.from_numpy(np.stack(ms)).to(dispositivo).transpose(1, 2)
            lg = rede(x)
            por_pergunta = (perda_fn(lg, y) * m).sum(dim=(0, 1)) / m.sum(dim=(0, 1)).clamp(min=1.0)
            perda = por_pergunta.sum()
            otim.zero_grad()
            perda.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otim.step()
            agenda.step()
            soma += perda.item()
        print("    época {}/{}: perda {:.3f} ({:.0f} s)".format(ep + 1, epocas, soma / passos_por_epoca,
                                                           time.time() - t0), flush=True)
    rede.to("cpu").eval()
    return ModeloSequencia(tipo, rede, norm, h)
