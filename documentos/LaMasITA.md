---
title: "LaMasITA — Technical Document"
subtitle: "More than a Hack 2027, Phase 1: Ball Possession (BAR-ATH)"
author:
  - Samuel Jales Terrinha Guimarães
  - Pedro Henrique Oliveira de Araújo
  - Nuno Prado de Medeiros Paulos
  - Arthur Augusto de Queiroz Gomes
  - Bryan Santos Monteiro
geometry: margin=2.2cm
fontsize: 10pt
---

Repository: `github.com/jalessoficial-commits/LaMasITA`. Every number below is taken from the READMEs
of the repository (`tracking_bola/`, `posse/`, `posse_ml/`, `diagnostico/`, `viewer/`), from the
notebooks, docstrings and commit notes of `analise/`, or from `evaluation/run_evaluation.py` and
`analise/casar_campeao.py` on the public 5 minutes.

## 1. Approach

The submission is produced by three layers, each replaceable and each validated on its own:

| layer | folder | what it does |
|---|---|---|
| **Ball** | `tracking_bola/` | our own ball track from the tactical video: a public football ball detector on every graded frame, a pitch mask, and a Viterbi track over the detections and the ATD ball |
| **Dead ball (D)** | `analise/` | a deterministic stoppage detector: each stoppage is an interval [start, restart], found from the referee's whistle (audio), the ball leaving the pitch, collective player speed and the Smart Tagging |
| **Possession (H/A)** | `posse_ml/` | a temporal convolutional network trained only on the seven open DFL matches, degraded to look like our tracking, decoded with a Viterbi that charges a price per change of possession |

**Why this shape.** Three measurements drove the design.

1. *The ball is the bottleneck of the input.* The ATD has a ball in 59.3% of the graded frames, and where
   it has one the median error is 15.7 px for a ball 8.2 px wide. We therefore rebuilt the ball from the
   video rather than interpolating the ATD.
2. *A false D costs more than a missed D.* With the ground-truth teams and no D at all, spellF1 on the
   public window is already ≈ 0.89, above what the baseline's own D gives (0.79): a false D splits a
   real possession in two and neither half matches. The dead-ball detector was therefore built for
   precision, as explicit stoppages with a confirmation step, and not as a per-frame classifier.
3. *Possession is a segmentation problem, not a per-frame one.* The four official baselines are one
   algorithm with four smoothing windows, and spellF1 moves by a factor of eleven between them. All our
   H/A decisions go through a decoder in which changing possession has a cost, so a change only happens
   when the following frames, added up, pay for it.

**No supervision from the challenge match.** The public ground truth has 17 possessions and 4 dead-ball
segments, too few to learn from. The possession network learns from the DFL open data, and the dead-ball
thresholds are fitted on weak labels derived from the Smart Tagging over the whole match (excluding the
public window ± 1 min). Section 5 states exactly what the public window was used for.

### 1.1 Ball (`tracking_bola/`)

No model was trained; only public weights are used. The detector is the Roboflow football ball detector
(YOLOv8, one class, from `github.com/roboflow/sports`), run locally on 640×640 tiles at native resolution
(downscaling the whole frame erases the ball). Each candidate is taken from pixels to the pitch through
the per-frame homography and discarded if it falls more than **2 m** outside the lines: most false alarms
were objects outside the pitch detected with high confidence, which temporal consistency cannot remove
because a static object is continuous.

A per-frame fusion with the ATD was the first tracker (detector alone if confidence ≥ 0.8 where the ATD has
no ball; where it has one, the detector refines it within 40 px, overrides it with confidence ≥ 0.8, or the
ATD is kept). It was chosen on 292 hand-annotated frames and measured on 93 new ones, never used to choose
anything:

| test set, 40 px tolerance | ATD | fusion |
|---|---|---|
| hit rate (79 frames with ball) | 53.2% | 67.1% |
| hit rate where the ATD has no ball (29) | 0% | 24.1% |
| median error on hits | 8.2 px | 0.4 px |
| false alarm (14 frames without ball) | 7.1% | 7.1% |
| F1 | 64.6% | 76.8% |

The possession layer uses a **Viterbi track** over the same candidates (`track_temporal.py`): the
lowest-cost sequence over each half, with cost 1 − confidence per candidate, 0.3 for the ATD ball, 0.5 per
frame without ball, 0.1 per metre between frames, jumps above 45 m/s forbidden and gaps of up to 12 frames
interpolated. On the test set it has a higher hit rate (79.7%) and a higher false-alarm rate (42.9%) than
the fusion: it accepts coherent sequences of medium-confidence detections that the fusion discards. Over
the whole match it has a ball in 83.5% of graded frames (fusion 72.8%, ATD 59.3%) and 34 gaps of 2 s or more
with the ball in play (fusion 59, ATD 51). Possession models built on it validated better than on the fusion,
on the DFL (spellF1 0.596 against 0.580) and on the public window (0.690 against 0.662, trees).
The annotated frames were used only to validate and test; none of them is an output of the system.

### 1.2 Dead ball (`analise/`)

**v1, a single readable rule.** A cascade of rules was fitted on weak labels. One node survived pruning:
*the mean player speed over the next 5 s is below 1.1 m/s*. Using future windows is legitimate because the
submission is offline. Rules on polarisation, chasing and restart formation were true but redundant with
speed. This rule finds the restart precisely (−0.4 to +3 s on the public window) and the start late (5–9 s):
players keep running after the whistle.

**v2, stoppages instead of frames** (`paradas.py`, `eventos.py`, `audio_apito.py`). Each end of a stoppage
has its own sensor. *Start:* the referee's whistle, detected in the audio of the tactical video as a
near-pure tone (≈ 3.8 kHz) standing out from the neighbouring bands, with the frequency calibrated on the
kick-offs marked by the Smart Tagging; or the ball leaving the pitch (it disappears after being within 2 m of a
line, or stays on the line). *Restart:* the Smart Tagging set piece plus a per-type offset calibrated
against the following collective acceleration, or that acceleration itself. A start only *opens a
candidate*; it becomes D only if confirmed (the v1 rule fires, a tagged restart of a compatible type
follows, players converge on the box after a ball over the goal line, or play brakes after a whistle). A
whistle without confirmation is advantage or crowd noise and is dropped. On the public window, D recall
went from 0.63 (v1) to 0.99 with precision ≈ 0.89, and the start error from 5–9 s to ≈ 1 s.

**v3–v5, the submitted detector.** v3 adds D from a goal to the kick-off and from the start of each half
to the kick-off (Smart Tagging), vetoes untagged stoppages in which the ball is visible and away from the
lines, and merges technical pauses. v4 moves the start of untagged stoppages back to a discarded strong
whistle or to where the ball disappeared, and cuts a stoppage when the ball is back deep in the pitch with
players running. v5 handles the goalkeeper: when the keeper catches the ball the ATD loses it and v1 marked
D, so a D of up to 25 s that starts with the ball within 3 m of the keeper inside the box, with no whistle
and no restart tag, becomes possession of the keeper's team.

### 1.3 Possession (`posse_ml/`)

**Features.** 100 columns per frame, none using player identity: nearest player of each team to the ball,
how many within 2 m and 5 m, where the ball will be in 0.5 s and 1 s, how long the ball has been unseen,
player pace, and the shape of each team in its own attacking direction (centre, spread, last line), plus
their 2-s trends. Team shape is what remains when the tracker loses the ball.

**Training data.** The seven open DFL matches carry ball status and team in possession per frame. Each
match is *degraded until it looks like ours*: noise and dropouts on players, aerial balls projected on the
ground as the tactical camera would, and ball gaps and jumps drawn from statistics **measured on our own
ball track** (fraction missing in play and in stoppages, gap-length histograms, jump rate). Three degraded
copies per match; two training sets, one with sides swapped so the model cannot learn that the home team
has more of the ball, averaged in log-odds.

**Model.** A non-causal TCN (6 residual blocks, dilations 1–32, width 64, ≈ 10 s of context each side) with
three outputs — *is the ball dead*, *is it the home team's*, *will the home team restart* — each trained
only on the frames where its question makes sense. It runs at 12.5 Hz on even and odd frames. The
submitted model is the **average of six seeds** (10, 11, 12, 20, 21, 22), 5 epochs each, in log-odds:
seeds differ by 0.002–0.007 on DFL validation but by 0.01–0.04 on the public window, so picking one seed
by the window would be picking by luck.

**Decoder.** A Viterbi with four states (alive-home, alive-away, dead-home-to-restart,
dead-away-to-restart) and a price per transition, chosen on leave-one-match-out validation on the DFL
(change 6, stoppage 12, restart 1). The dead ball is used in *rigid* mode: alive/dead is fixed by the
detector and the network only decides whose ball it is and who restarts.

**Final assembly** (`analise/casar_campeao.py`). The network is decoded with the v2 dead ball, then the v5
dead ball is laid over it: frames dead in v5 are D; frames alive in v5 that the first pass had as D take
the nearest H/A in time; the goalkeeper segments of v5 take the keeper's team; and an H/A possession
shorter than 1 s squeezed between two possessions of the other team, or between a D and one, is absorbed.

## 2. Possession rules implemented

No possession rule is hand-written for H/A; they follow from the price per change in the decoder.

- **A change needs sustained evidence.** Possession passes to the other team only when the following
  frames, added up, favour it enough to pay for the change. While the ball travels or is unseen, frames
  weigh little either way and possession stays where it was.
- **Loose balls and challenges do not change possession.** With players of both teams at the ball each
  frame is near a tie. A deflection or a touch that does not keep the ball gives too few frames to pay for a
  change; if the opponent comes away with the ball, the following frames pay.
- **Dead ball.** D runs from the event that stops play (whistle, or ball out) to the restart; from a goal
  to the kick-off; and before each half's kick-off. During a stoppage nobody has the ball, but the dead
  state already carries who will restart, and possession after the restart is that team's.
- **Goalkeeper holding the ball** is possession of the keeper's team, not a dead ball. Because the ATD
  goalkeeper slots swap identity, a keeper slot is assigned to the team defending the goal on its side of
  the pitch.
- **Possessions under 1 s** between two possessions of the other team are not kept.
- **X** outside the two halves; the system never abstains on a graded frame.

## 3. Assets and methods: used, tried and dropped

**Provided assets.** *Tactical video:* ball detection and audio (whistle). *ATD:* all players; the ATD ball
as one candidate in the ball track and as the ball of the dead-ball start events (our tracker's ball
enters only the v3 goal and live-ball signals). *Homography:* ball from
pixels to pitch, pitch mask, camera position for the DFL degradation. *Smart Tagging:* set pieces, goals
and kick-offs in the dead-ball detector; weak labels; whistle frequency calibration. *Public ground truth:*
see Section 5. *Broadcast video:* tried and dropped — aligned to the tactical video to tell live shots
from replays as evidence of dead ball; that evidence is not used in the rigid mode of the submission.

**External.** Roboflow ball detector weights (public); Sportec Open DFL dataset (Bassek et al., 2025,
CC-BY 4.0), the only training data of the possession network.

Tried and dropped, with the number that decided it (possession rows: spellF1 on the public window):

| what | result | why dropped |
|---|---|---|
| YOLOv8 (COCO, `sports ball`), no training | did not beat the ATD | COCO has almost no small balls in open play |
| FootAndBall, no training | ≈ 1% hit rate | barely responds to a ball this small |
| WASB-SBDT fused with the ATD | F1 60.6% vs 73.3% (validation) | many far errors; Roboflow recovers twice as much where the ATD is blind |
| Kalman ball track | lowest F1; worse where the ATD has a ball | the track sticks to wrong objects |
| CART and 2-state HMM for dead ball | CART relearned the weak-label rule; HMM stuck in D | uncalibrated probabilities |
| Heuristic possession (`posse/`: nearest player, touch as a necessary condition) | 0.586 | predicts 11 possessions against 17; cannot tell a short possession from a touch the GT ignores |
| Gradient-boosted trees (HGB, XGBoost, LightGBM, CatBoost) | 0.690 | predict 12 of 17 possessions even with perfect dead ball |
| GRU; transformer | 0.640–0.679; 0.658–0.687 | below TCN and LSTM |
| GRU + TCN ensemble | 0.658 | both err on the same plays |
| TCN / LSTM with 10 or 20 epochs | 0.636–0.698 | training longer transfers worse to our match |
| Training on Metrica open matches, alone or with DFL | 0.648–0.693; 0.686–0.730 | their ball is rebuilt from events; fewer possessions, more variance |
| The network's own dead ball instead of the detector | 0.678 (TCN, seed 0) | the detector is better at D |
| Soft mode (detector as one more evidence) | ties with rigid | needs four weights fitted on the public window |

## 4. Results on the public 5 minutes

| version | spellF1 | macroF1 | possessions (GT 17) |
|---|---|---|---|
| official baseline, `smooth50` / `smooth125` | 0.326 / 0.424 | 0.621 / 0.581 | 26 / 16 |
| baseline H/A + our dead ball (v2) | 0.485 | 0.893 | 16 |
| heuristic H/A (`posse/`) + dead ball v2 | 0.586 | 0.920 | 11 |
| trees (HGB) + dead ball v2, rigid | 0.690 | 0.931 | 12 |
| TCN, 5 epochs, single seeds (mean ± sd of six) | 0.724 ± 0.009 | 0.937 | — |
| TCN, six-seed average, dead ball v2 | 0.739 | 0.932 | 16 |
| **submission: + dead ball v5, possessions < 1 s absorbed** | **0.739** | **0.932** | 16 |

The last step does not move the public window, whose four dead-ball segments were already right. It was
adopted for the rest of the match, checked on two 5-minute blocks labelled by hand in our viewer and
used for validation only: there, dead ball v4 with the short possessions absorbed took spellF1 from 0.550
to 0.687 and macroF1 from 0.832 to 0.870, and v5 took macroF1 to 0.889.

On DFL leave-one-match-out validation the six-seed TCN scores spellF1 0.621 and macroF1 0.916. The DFL
labels change possession more often than the challenge's (≈ 5 possessions per minute against 3.4), so this
is not the same scale as the public window.

## 5. What the public ground truth and hand labels were used for

- **Never used for training.** No weight of any model was fitted on the challenge match.
- **Used to choose between candidate configurations** (network type, epochs, training data, versions of the
  dead-ball detector). For the networks the rule was fixed beforehand: mean spellF1 over seeds, differences
  under 0.03 are a tie, broken by macroF1. This makes the public-window number somewhat optimistic.
- The values that `posse_ml/calibrar.py` fits on the public window (weight of the broadcast evidence,
  advance of stoppage borders) are **not used by the submission**: in rigid mode they cannot change the
  decoding (`posse_ml/testes/test_rigido.py`).
- **Small hand-labelled samples, validation only** (allowed by RULES.md): 292 + 93 frames with the ball
  position; start and end of a sample of stoppages, and verdicts on dead-ball clips of the first half, with
  which some v4 thresholds were revised; two
  5-minute blocks of possession. None is in the repository and the submitted file does not depend on them.
- No script sends match data off the machine; the only network access downloads public weights and open datasets.

## 6. Limitations

- **The full-match score is unknown to us.** The public window has 17 possessions: one more or one fewer
  moves spellF1 by ≈ 0.06, and the scorer's bootstrap gives ≈ 0.45–0.85 for every version of `posse_ml`.
- **Possession borders.** The network finds the right number of possessions but not the exact frame of
  each change: for the six-seed TCN, transition F1 is 0.72 at ± 25 frames and 0.31 at ± 12.
- **The ball is still missing in about one frame in six,** and the Viterbi track trades hits for false
  alarms (42.9% on 14 test frames without ball). An invented ball next to the wrong player is the costlier error.
- **High balls.** The homography holds for the ground: a ball in the air is projected farther than it is,
  can be cut by the pitch mask, and looks like a touch with nobody near.
- **Domain.** The ball detector was trained on Bundesliga broadcast video, not on a tactical camera, and was
  not adapted. The possession network never saw this match or this labelling convention.
- **Dead ball is accepted as given.** A wrong D from the detector goes straight to the submission. Its start
  is the weak end where there is no whistle, and its later versions were tuned on few examples of one match.
- **Goalkeepers.** If a keeper's identity moved durably to a third ATD slot, that stretch is not corrected.
- **Small validation samples everywhere:** 93 test frames for the ball, 4 dead-ball segments and 17
  possessions for the rest.

With more time: handle the aerial ball explicitly; sharpen the frame of each change of possession;
calibrate the dead-ball detector on more matches.

## 7. Reproduction

Python 3.13; the match data in `data/` as described in DATA.md.

```bash
pip install -r requirements-submissao.txt
gdown -O tracking_bola/pesos/football-ball-detection.pt \
      "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
python gerar_submissao.py     # writes LaMasITA.csv and validates it
```

`gerar_submissao.py` runs, in order: ball detection, mask and fusion; export of the ball; Viterbi track;
features and whistle; dead ball v2 and v5; the TCN (trained weights in `modelo/campeao_tcn/`); the final
assembly; and the format check. Ball detection took about 27 h on a laptop GPU (RTX 3050, 6 GB) and resumes
if interrupted; every other step takes minutes. The network always predicts on CPU so that the output does
not depend on the machine. Retraining is not needed, but is reproducible with `python -m posse_ml.baixar_dfl`
and the commands in `posse_ml/README.md`.

## References

Bassek, Rein, Weber, Memmert (2025), *An integrated dataset of spatiotemporal and event data in elite
soccer*, Scientific Data 12:195. · Roboflow, *sports*, github.com/roboflow/sports. · Tarashima et al.,
*WASB-SBDT*, BMVC 2023. · Komorowski et al., *FootAndBall*, VISAPP 2020. · Metrica Sports, *sample-data*.
