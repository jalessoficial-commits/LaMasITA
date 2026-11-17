# Submissão da equipe LaMasITA

> O restante deste README, a partir de "More than a Hack 2027", é o texto original do desafio.

**O que a submissão faz.** Para cada frame, decide entre H, A, D e X em três camadas, cada uma com a sua pasta:

| camada | pasta | o que faz |
|---|---|---|
| bola | `tracking_bola/` | detector de bola do Roboflow (YOLOv8, pesos prontos, inferência local) em todos os frames avaliados, máscara do campo, fusão com o ATD e rastro por Viterbi |
| bola morta | `analise/` | detector de paradas (versão v5): árvore treinada com rótulos fracos do próprio ATD, apito no áudio, Smart Tagging e regras de reinício |
| posse H/A | `posse_ml/`, `modelo/campeao_tcn/` | TCN treinada só nos sete jogos abertos da DFL, média de seis sementes, decodificada por Viterbi com a bola morta fixada pelo detector (modo rígido) |

No GT público (5 minutos), o arquivo final tem spellF1 0,739.

**Uso do GT público.** As redes foram treinadas só na DFL e nenhum peso foi ajustado no jogo do desafio. A janela
pública de 5 minutos foi usada para escolher entre configurações candidatas (rede, épocas, dados de treino, versões
do detector de bola morta), sem treinar nem ajustar pesos nela. Os números calculados na janela pelo
`posse_ml/calibrar.py` (`calibracao*.json`) não entram na submissão: no modo rígido eles não mudariam a decodificação,
e `posse_ml/prever.py` não os usa. Os parâmetros da fusão da bola e do detector de bola morta foram escolhidos à mão,
comparando as configurações no viewer. Pequenas amostras rotuladas à mão (frames com a bola, vereditos de trechos
do 1º tempo, dois blocos de posse) serviram só para conferir que o modelo funcionava fora do GT público, sem treinar nem escolher nada; esses rótulos
não estão no repositório, e o arquivo não depende deles.

## Reproduzir o arquivo enviado

1. **Python 3.13** e as dependências:

   ```bash
   py -3.13 -m pip install -r requirements-submissao.txt
   ```

2. **Dados do desafio** em `data/` (ou `MTAH_DATA_DIR`), no layout do [DATA.md](DATA.md).

3. **Pesos do detector de bola** (do repositório [roboflow/sports](https://github.com/roboflow/sports), `examples/soccer/setup.sh`):

   ```bash
   gdown -O tracking_bola/pesos/football-ball-detection.pt "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
   ```

4. **Um comando**, que roda as etapas na ordem e grava `LaMasITA.csv` na raiz, já conferido com o
   `evaluation/validate_submission.py`:

   ```bash
   py -3.13 gerar_submissao.py
   ```

   No Linux ou no Mac, use `python3.13` no lugar de `py -3.13`.

   A detecção da bola leva horas (usa a GPU se o PyTorch tiver CUDA) e pode ser interrompida: rodar de novo
   continua de onde parou. As outras etapas levam minutos. As etapas e as saídas intermediárias estão descritas no
   cabeçalho de `gerar_submissao.py`. Tudo o que é derivado dos dados fica fora do Git.

**O que foi conferido.** Refazendo do rastro da bola em diante (`gerar_submissao.py --conferir`), o arquivo sai
idêntico byte a byte (mesmo sha256) na máquina que gerou a submissão. A detecção da bola, que levou cerca de 27 h
numa GPU, não foi refeita: uma rede em GPU pode variar levemente com a placa e o driver, e uma detecção perto do
limiar de confiança que mude de lado se propaga a alguns frames do arquivo final. O pipeline usa só o vídeo
tático; o vídeo de transmissão não é lido.

Os detalhes de cada camada, com os testes e o que foi descartado, estão em `tracking_bola/README.md`,
`analise/` (`paradas.py`, `avaliar_v3.py`, `casar_campeao.py`) e `posse_ml/README.md`.

---

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play                                                                                           |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone
```

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
