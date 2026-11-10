# posse/

Rotulagem de posse (H ou A) dentro dos trechos de bola viva. A bola morta (D) vem do detector de
paradas do Nuno (`analise/03_detector_paradas.ipynb`, arquivo `analise/out/paradas_D.csv`); esta
pasta só decide **de quem é a bola** quando ela está viva. A seção 2 mostra como gerar a submissão do
jogo inteiro com as duas partes.

| script | o que faz |
|---|---|
| `restart_owner.py` | o dono no reinício: todo reinício tem um dono claro (CHALLENGE.md), então cada trecho vivo começa com um time conhecido |
| `predictor.py` | o preditor completo: parte do dono do reinício, decide quando a posse troca dentro do trecho e gera a submissão |
| `diagnostico_toques.py` | compara os eventos da bola perto das trocas do GT público com os toques do adversário que o GT ignora |
| `comparar_bolas.py` | roda o preditor com cada bola (fusão, Viterbi, Kalman), com e sem filtro de pulos, e mostra uma tabela |
| `varrer_portao.py` | roda o preditor com várias janelas do portão (frames antes e depois de cada evento) e mostra uma tabela |

Nenhum script envia dados para fora da máquina. Tudo o que é derivado dos dados do jogo (tabelas por
frame, camadas para o viewer) fica em `posse/out/`, que não pode ir para o git: o repositório é
público.

---

## 1. predictor.py

### 1.1 Visão geral

Dentro de cada trecho vivo, um Viterbi com dois estados, H e A, escolhe a sequência de menor custo:

```latex
\hat{y}_{1:T} = \arg\min_{y_{1:T}} \left[ \sum_{t=1}^{T} c_t(y_t) + C \sum_{t=2}^{T} [y_t \ne y_{t-1}] \right]
\quad \text{sujeito a: troca para } X \text{ em } t \text{ só se } \mathrm{liberada}(X, t)
```

- **C = 1** (`SWITCH_COST`) por troca de time.
- **Início:** o trecho começa no dono do reinício (`restart_owner.py`); começar no outro time custa 5,
  ou 1 quando o dono saiu com pouca evidência.
- **Custo por frame** $c_t(X)$: a evidência, naquele frame, a favor do **outro** time (seção 1.4).
- **Portão** (seção 1.5): a troca para X só é possível perto de um toque de X. **O toque é condição
  necessária, mas não suficiente**: ele libera a troca, e a evidência depois dele decide se ela
  acontece.

Onde não há bola, não há evidência e o Viterbi mantém a posse. É o mesmo efeito do X no spellF1, mas
sem o custo do X no macroF1.

### 1.2 Bola

A do CSV em `--ball`, ou a do ATD sem ele. O preditor aceita três formatos (`--ball-units`): `norm`
(0 a 1, como o ATD e o `BAR-ATH_tactical_ball_tracker.csv`), `m` (metros) e `px` (pixels do vídeo,
como a saída do `tracking_bola/track_temporal.py`, convertida pela inversa da homografia de cada
frame, como no `export_ball.py`).

**Bola campeã:** a do Viterbi do tracking (`tracking_bola/out/viterbi_bola.csv`, `--ball-units px`),
com o filtro de pulos desligado (`--no-despike`). Ver a seção 1.9.

Duas decisões vêm do `tracking_bola/README.md`:

- **Sem interpolação.** O Kalman e o Viterbi do passo 15 não filtraram erros, e a decisão foi deixar a
  camada de posse atravessar as lacunas da bola (seção 6). Além disso, uma reta entre duas medidas
  ruidosas cria quebras de velocidade nas pontas, que virariam toques falsos.
- **Filtro de pulos** (desligável com `--no-despike`). O modelo final do tracking acerta 67,1% dos
  frames com bola no teste, e cerca de 1 em 10 posições que ele dá está errada, em geral um pulo para
  outro objeto por poucos frames. Cada pulo cria uma quebra de velocidade, ou seja, um toque falso.
  Cada posição é comparada com a mediana, coordenada a coordenada, das posições dos 5 frames antes e
  5 depois; se estiver a mais de 3 m, é descartada. A mediana acompanha a bola em linha reta e na
  quebra de direção de um toque real; um pulo de até 5 frames não a arrasta. Para a bola do Viterbi,
  que já impede saltos impossíveis, o filtro é indiferente, e por isso fica desligado.

Depois do filtro, a posição passa por uma média móvel de 5 frames.

### 1.3 Time de cada jogador: correção dos goleiros

O time de cada slot vem do XML do ATD e é fixo, mas os slots dos goleiros trocam de pessoa: em alguns
trechos o slot do goleiro do Athletic passa a seguir o goleiro do Barcelona, com o time errado, e o
slot do goleiro do Barcelona fica parado como um "fantasma" (`diagnostico/README.md`, passos 2 e 4).
No lance 15652–15686 do GT público, o goleiro do Athletic com a bola aparecia como jogador do Barça, e
o preditor inventava uma posse curta do Barcelona.

Regra (desligável com `--no-gk-fix`):

1. **Quem são os goleiros.** Em cada time, o slot que mais vezes é o jogador do time mais perto do
   próprio gol (mesmo critério do `diagnostico/verificar_goleiro.py`; só slots presentes em pelo
   menos 1.000 frames). Neste jogo: slots 24 e 46. Se nenhum slot for o mais recuado do time em pelo
   menos metade dos frames, esse time fica sem correção.
2. **Time pelo lado do campo.** Em cada frame, o slot de goleiro recebe o time que defende o gol do
   lado do campo em que ele está. O lado de cada time vem do kickoff de cada tempo (frames 11185 e
   113742), quando todos estão no próprio campo.
3. **Os outros slots mantêm o time do ATD.** Um jogador de linha mais recuado que o goleiro (lance na
   pequena área, goleiro adiantado) não é afetado.

Uma versão anterior tratava como goleiro o jogador mais recuado de cada grande área. Ela resolveu o
lance 15652, mas também trocou o time de jogadores de linha (slots 21, 41, 45 e 4, algumas centenas de
frames cada). A regra atual só mexe nos dois slots de goleiro e dá o mesmo resultado no GT público
(seção 1.9). Limitação: se a identidade do goleiro passar de forma duradoura para um terceiro slot,
esse trecho não é corrigido; pelo diagnóstico, nenhum outro slot passa de 7% como mais recuado, e esse
slot (42) tem perfil de zagueiro (9% do tempo na área, 43 m de mediana até a linha de fundo).

O terminal imprime os slots escolhidos e em quantos frames o time de cada um foi corrigido.

### 1.4 Evidência por frame

| fonte | o que é | peso |
|---|---|---|
| **Toque estrito** | pico local da mudança do vetor velocidade da bola (média dos 4 frames antes contra a dos 4 depois, pelo menos 3 velocidades válidas de cada lado), acima de 3× o ruído típico da bola (mínimo 4 m/s), fora de voo (seção 1.5); vai para o time do jogador mais perto, se ele estiver a até 2 m e 0,75 m mais perto que o adversário | 0,5 com o portão (1,5 sem) |
| **Proximidade** | a cada frame com bola, o time mais perto: `0,08 · exp(−(d/2,5)²) · tanh(folga/1)`, com *d* a distância do mais perto e a folga sobre o adversário em metros | até 0,08 por frame |
| **Último toque** | enquanto não há evidência, o time do último toque estrito | 0,01 por frame (desempate) |

- **Toque como mudança de velocidade**: Vidal-Codina et al. (2022), implementado na DataBallPy, só
  aceitam ganho de bola se ela muda de direção (≥ 10°) ou de velocidade (≥ 5 m/s) perto do jogador;
  o vetor velocidade junta os dois critérios. O raio de 2 m é maior que o 1,5 m da DataBallPy porque o
  tracking automático tem mais ruído.
- **Bola disputada não conta**: com os dois times a menos de 0,75 m de diferença, o toque não vale
  para ninguém. É a regra do desafio para bola dividida (a posse fica com quem tinha).
- **Limiar adaptativo**: a maioria dos frames vivos não tem toque, então a mediana da mudança de
  velocidade mede o ruído da própria bola, e o limiar se ajusta à bola usada sem olhar rótulo nenhum.
- **Peso 0,5 com o portão**: abaixo do custo da troca (1), então um toque sozinho não troca a posse;
  ele precisa da proximidade do novo time depois (~0,5 s de domínio claro) ou de outro toque.
- **Último toque**: faz a troca cair no frame do toque. Nas trocas do GT público, a troca acontece no
  momento do toque (por exemplo, a 2 e a 4 frames da recepção em 11351 e 15766).

**Smart Tagging de fases** (`--tag-weight`, desligado por padrão): cada fase com time soma evidência
por frame para o time com a bola; nas fases de defesa (`DEFENDING IN ...`, `DEFENSIVE TRANSITION`) o
time da tag é o que defende, e a evidência vai para o outro.

### 1.5 Portão: o toque como condição necessária

A troca **para o time X** só é permitida a até ±0,6 s (15 frames; até a versão anterior, 25) de um
evento que a libere
(desligável com `--no-gate`, que volta à política anterior, com toque de peso 1,5; a janela muda com
`--gate-window` e `--gate-before`, e o `varrer_portao.py` compara várias):

1. **Eventos permissivos.** Picos da mudança de velocidade acima de 2× o ruído (mínimo 3 m/s), mais
   baixos que os do toque estrito, porque aqui o evento só libera a troca. Cada evento libera a troca
   para todo time que tenha **algum** jogador a até 3 m da bola, mesmo sem ser o mais perto. Motivo:
   em 11351 e 15766 o GT troca para o Barcelona na recepção, mas o recebedor estava a 2–3 m da bola no
   tracking, com um marcador do Athletic mais perto; com a regra do "mais perto", essas duas trocas
   ficariam impossíveis.
2. **Sem bola em voo.** Bola rápida antes e depois (≥ 12 m/s) e quase sem mudar de direção (< 30°) é
   bola em voo, não toque. A homografia vale para o chão: a projeção de uma bola alta não anda em linha
   reta com velocidade constante e parece acelerar ou curvar sem ninguém tocá-la (lance 12757). Esse
   filtro vale para os eventos do portão e para os toques estritos.
3. **Reaparecimento da bola.** Se a bola some por 10 frames ou mais e reaparece com um time claramente
   nela (critério do toque estrito, nos 5 primeiros frames), a troca para esse time fica liberada na
   lacuna inteira e até 1 s depois: o toque aconteceu no escuro.

O terminal mostra quantos eventos e reaparecimentos liberaram trocas e em que fração dos frames vivos
a troca está liberada para cada time. Na tabela de trocas do GT, a coluna `liberada` diz se o portão
deixava cada troca verdadeira acontecer.

### 1.6 Opções experimentais (desligadas por padrão)

Cada uma foi testada no GT público sobre a configuração campeã e nenhuma melhorou. Ficam no código
para comparação.

| opção | ideia | resultado no GT público |
|---|---|---|
| `--anchor` | o Viterbi decide **se** a posse troca e o toque decide **quando**: cada troca prevista vai para o toque do novo time mais próximo | piorou: adiantou trocas que já saíam cedo |
| `--prox-control` | a proximidade só conta com a bola sob controle (até 7 m/s), não em trânsito | piorou: atrasou trocas demais |
| `--touch-after` | o toque vai para quem sai com a bola nos ~0,5 s seguintes (12 frames, média até 2 m), e não para o mais perto no instante; motivado pela recepção sob marcação de 11349 | piorou no resto do trecho |
| `--aerial` | toques no ar (cabeça, peito) | não melhorou |

**Toques aéreos (`--aerial`).** A bola alta é vista no chão no ponto G, mas pode estar em qualquer
ponto do raio que liga a câmera a G.

1. **Câmera recuperada da homografia.** O centro óptico é suposto no meio da imagem (1920×1080), a
   distância focal sai das duas restrições de ortogonalidade da rotação, e a posição é C = −Rᵀt, com
   altura C_z > 0. Neste jogo: x ≈ 53,6 m, y ≈ 115,7 m, altura ≈ 30 m (variação de 1–2 m entre
   frames) e distância focal ≈ 2449 px. Frames com altura fora de 3–120 m são descartados.
2. **Distância de cada jogador ao raio.** Até 2,5 m de altura (cabeçada com salto), o raio cobre no
   chão o segmento de G em direção à projeção da câmera, de comprimento 2,5 / C_z · |C_h − G|. A
   distância de um jogador ao raio é a distância dele, no chão, a esse segmento. Toque estrito a até
   1 m do raio; liberação do portão a até 1,5 m.
3. **Quando vale.** Só quando ninguém está a até 2 m da bola no chão e nenhuma posição da bola pula
   nos 5 frames em volta. Sem essa guarda, no simulador sem filtro de pulos, os pulos do tracker
   geravam toques aéreos falsos (78 em 507 toques); com ela, nenhum.

### 1.7 Como rodar

Teste nos 5 minutos públicos, com as fronteiras de bola morta do GT (`--live-ha`), na configuração
campeã:

```powershell
python posse/predictor.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball tracking_bola/out/viterbi_bola.csv --ball-units px --no-despike
```

Explicar um lance frame a frame (por que a posse trocou ou não):

```powershell
python posse/predictor.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball tracking_bola/out/viterbi_bola.csv --ball-units px --no-despike --explain 11540 11640
```

Jogo inteiro, com a bola morta do Nuno: seção 2.

| argumento | o que faz |
|---|---|
| `--dead` | arquivo de bola morta (obrigatório): `frame,morta` (como o `paradas_D.csv`) ou `frame,label` com D |
| `--live-ha` | só H e A contam como vivo (para usar o GT público como `--dead`) |
| `--ball`, `--ball-cols`, `--ball-units`, `--ball-frame-base` | a bola e o seu formato (seção 1.2) |
| `--no-despike` | desliga o filtro de pulos |
| `--no-gk-fix` | desliga a correção dos goleiros |
| `--no-gate` | desliga o portão (política anterior, toque com peso 1,5) |
| `--gate-window`, `--gate-before` | frames depois e antes de cada evento em que o portão libera a troca (padrão 15 e 15; sem `--gate-before`, os dois iguais) |
| `--touch-weight` | peso do toque estrito (padrão 0,5 com o portão, 1,5 sem) |
| `--tag-weight` | peso das fases do Smart Tagging (padrão 0) |
| `--explain INICIO FIM` | explica o intervalo frame a frame (seção 1.8) |
| `--anchor`, `--prox-control`, `--touch-after`, `--aerial` | opções experimentais (seção 1.6) |
| `--out` | arquivo da submissão (padrão `posse/out/posse.csv`); é sobrescrito a cada execução |

O ATD é lido inteiro (~1–2 min).

### 1.8 Saídas

- **Submissão**: D na bola morta, H/A nos trechos vivos, X fora das metades.
- **`posse/out/toques.csv`**: camada para o viewer (**➕ Adicionar tracking**) com a bola em cada toque
  estrito, na cor do time. Junto com a submissão em **🎯 Carregar predição**, mostra por que o preditor
  trocou (ou não) em cada lance. É sobrescrito a cada execução, então corresponde sempre à última: uma
  rodada com `--live-ha` deixa nele só os toques da janela pública.
- **Terminal, com o GT público**: spellF1 e macroF1 do avaliador oficial; cada troca do GT com a troca
  prevista correspondente (até 2 s, no mesmo sentido), o desvio em frames e se estava liberada; e as
  trocas previstas sem par no GT.
- **Terminal, no jogo inteiro**: número de trechos vivos, de trocas previstas, de posses e a duração
  média delas.
- **Com `--explain`**: (1) cada pico da mudança de velocidade no intervalo, com as velocidades antes e
  depois, o ângulo, a distância de cada time à bola no instante e nos ~0,5 s seguintes, se é voo e em
  que virou (toque estrito de quem, ou só evento do portão); (2) de 5 em 5 frames, o rótulo previsto,
  a evidência somada para H e para A, as distâncias médias de cada time, a fração de frames com bola e
  se a troca estava liberada para cada time.

### 1.9 Resultados

#### Com as fronteiras de bola morta do GT

As rodadas desta parte usam as fronteiras de bola morta do próprio GT (`--live-ha`), o que infla os
números, sobretudo o macroF1 (todo D sai certo): **não se comparam com os baselines oficiais**. São 17
posses e 12 trocas de posse com a bola viva; uma posse a mais ou a menos move o spellF1 em ~6 pontos.
Da quinta linha em diante, cada linha acrescenta uma mudança à configuração da anterior (bola do
Viterbi, filtro desligado).

| configuração | spellF1 | macroF1 | posses (GT 17) | trocas inventadas |
|---|---|---|---|---|
| fusão ATD + Roboflow, filtro ligado, toque 1,5 | 0,533 | 0,939 | 22 | 10 |
| fusão, filtro desligado | 0,435 | 0,902 | 29 | 17 |
| Viterbi do tracking, filtro ligado | 0,544 | 0,934 | 22 | 10 |
| Viterbi do tracking, filtro desligado | 0,550 | 0,943 | 23 | 11 |
| + goleiro (versão "mais recuado da área") | 0,568 | 0,948 | 21 | 9 |
| + portão (toque necessário, peso 0,5) | 0,686 | 0,960 | 11 | 2 |
| goleiro atual (slots 24 e 46 pelo lado do campo) no lugar do anterior | 0,686 | 0,960 | 11 | 2 |
| janela do portão de 15 frames (era 25) | **0,686** | **0,962** | 11 | 2 |

- **O filtro de pulos é necessário para a fusão** (−10 pontos sem ele) e indiferente para o Viterbi,
  que já impede saltos impossíveis. Fusão com filtro e Viterbi empatam dentro do ruído.
- **O Kalman do tracking** ficou abaixo dos dois. Com o goleiro atual e o portão, a comparação das
  bolas (`comparar_bolas.py`) manteve o Viterbi sem filtro na frente.
- **A correção do goleiro** tirou exatamente o par falso 15647/15694, sem mudar mais nada.
- **A linha "+ portão" mudou duas coisas de uma vez**: ligou o portão e baixou o peso do toque de
  1,5 para 0,5. Medidas separadas (seção 6), quase toda a queda das trocas inventadas vem do peso:

  | | spellF1, D do GT | inventadas | spellF1, D do Nuno | inventadas |
  |---|---|---|---|---|
  | sem portão, peso 1,5 | 0,568 | 9 | 0,541 | 8 |
  | sem portão, peso 0,5 | 0,621 | 3 | 0,557 | 3 |
  | com portão, peso 0,5 | 0,686 | 2 | 0,571 | 3 |

  O portão em si tira uma troca inventada com o D do GT (11461 e 13650 são as que sobram) e quase
  nada com o do Nuno, sem bloquear nenhuma troca verdadeira. Com a janela larga ele é permissivo (com
  25 frames, a troca fica liberada em ~73% dos frames vivos para H e ~56% para A), e de 20 a 100
  frames o resultado é idêntico: quem decide é a suficiência. O ganho que sobra do portão é de tempo:
  com 15 frames, ele impede que um evento cedo libere a troca muito antes do toque (seção 6).
- **O preditor é conservador**: 6 trocas previstas, 4 certas, e perdeu todas as posses curtas
  (11553→11624, 15346→15365, 15766→15799). Elas estavam liberadas pelo portão; quem barrou foi a
  suficiência (seção 1.10).

**Peso do toque estrito** (`--touch-weight`, configuração campeã; "certas" = trocas do GT acertadas até
2 s e no mesmo sentido, de 12):

| peso | spellF1 | certas | perdidas | inventadas |
|---|---|---|---|---|
| **0,5** (padrão) | **0,686** | 4 | 8 | 2 |
| 0,75 | 0,680 | 6 | 6 | 2 |
| 1,0 | 0,680 | 6 | 6 | 2 |
| 1,5 | 0,633 | 7 | 5 | 7 |

Até 1,0, o peso maior recupera duas trocas sem inventar nenhuma; em 1,5, recupera mais uma e inventa
cinco. 0,5 e 0,75 empatam dentro do ruído; o padrão fica em 0,5 até as janelas de validação (seção
1.13) desempatarem.

#### Jogo inteiro, com a bola morta do Nuno

Submissão `evaluation/results/posse_v1.csv` (seção 2), pontuada pelo `run_evaluation.py` no GT público,
ao lado das submissões que já estavam na pasta:

| submissão | spellF1 | macroF1 | posses (GT 17) | duração média (GT 14,2 s) |
|---|---|---|---|---|
| **posse_v2** (janela do portão de 15 frames) | **0,586** | **0,920** | 11 | — |
| posse_v1 (este preditor + paradas do Nuno, janela de 25) | 0,571 | 0,919 | 11 | 21,3 s |
| paradas_smooth125 (paradas do Nuno + time do baseline) | 0,485 | 0,893 | 16 | 14,6 s |
| baseline_smooth125 | 0,424 | 0,581 | 16 | 18,5 s |
| paradas_smooth50 | 0,421 | 0,879 | 22 | 10,6 s |
| baseline_smooth50 | 0,326 | 0,621 | 26 | 10,8 s |

- **O ganho sobre o `paradas_smooth125` é todo do time.** A bola morta é a mesma nos dois (F1 de D
  0,9377 e a mesma linha D na matriz de confusão); o F1 de H sobe de 0,887 para 0,916 e o de A, de
  0,856 para 0,904.
- **A queda de 0,686 (fronteiras do GT) para 0,571 é o custo das fronteiras imperfeitas.** 185 frames
  de bola viva (124 do Barça e 61 do Athletic) viram D e cortam as pontas das posses. Com
  sobreposição exigida de 30–40%, 10 das 17 posses batem (spellF1 0,714); com 60–70%, só 6 (0,429).
  O acerto de viva/morta é alto (F1 0,984), mas cada borda pesa.
- **Trocas (posse_v1):** F1 de 0,514 a ±25 frames, com as trocas saindo em média 11 frames
  adiantadas. No posse_v2, o desvio mediano das trocas certas vai de −19 para −3 frames, e uma posse
  passa de um limiar de sobreposição a mais (+0,014); as trocas certas, perdidas e inventadas não
  mudam. As análises abaixo são do posse_v1.
- **Cuidado:** o detector do Nuno e este preditor foram ajustados nesses mesmos 5 minutos, então a
  nota no GT privado deve ser menor. Com 17 posses, uma posse move ~0,07 no spellF1; a faixa de ruído
  real sai de `python evaluation/run_evaluation.py --bootstrap 1000`.

**O jogo inteiro é coerente** (`validate_submission.py`): 199.319 linhas válidas; H 60.394, A 43.170,
D 55.293 e X 40.462 (exatamente os frames fora dos tempos, 199.319 − 158.857); 304 posses de 13,6 s em
média, perto do ritmo do GT público; Barça com 58% do tempo de bola viva; bola morta em 35% do tempo
avaliado. Uma coisa não fecha: na janela pública o preditor prevê **menos** posses que o GT (11
contra 17), mas no jogo inteiro o ritmo bate com o do GT. Ou o trecho público é mais picotado que a
média, ou em outras partes o preditor inventa trocas que compensam as que perde. Só mais gabarito
separa as duas coisas.

### 1.10 Limitações conhecidas

- **Posses curtas são quase impossíveis.** Prever uma posse curta no meio de outra exige duas trocas
  (custo 2), e a evidência por frame é pequena: no máximo 0,08 de proximidade (o normal é 0,03–0,06)
  e 0,5 por toque estrito. Mesmo com domínio perfeito e nenhuma evidência do outro time, são precisos
  uns 20 frames só para empatar; abaixo de 20–25 frames, a posse não sai. Exemplos: 15346→15365 (19
  frames, no máximo 19 × 0,08 + 0,5 ≈ 2,0) e 15766→15799 (33 frames).
- **Toques que o GT ignora partem posses.** Na posse do Athletic de 11553 a 11624 (71 frames), um
  jogador do Barça toca a bola a 0,1 m dela em 11580, e o GT mantém a posse com o Athletic. Para o
  preditor, esse toque é evidência forte para o Barça, e o trecho do Athletic que sobra (~27 frames)
  não paga as duas trocas. O preditor não tem como separar uma posse curta de verdade de um toque que
  o GT ignora: os dois são um toque do outro time seguido de alguns frames perto da bola. Afrouxar o
  critério deixa passar os dois (peso 1,5: 7 trocas inventadas), e inventar custa ~1,5× mais que
  omitir, porque a posse inventada parte uma posse longa ao meio.
- **Trocas adiantadas (em parte resolvido).** Com a janela de 25 frames, as trocas certas saíam cedo:
  12665 (−21 frames), 13785 (−19) e 17293 (−23). Um evento cedo (o novo time a até 3 m da bola antes
  de ganhá-la) liberava a troca, e a proximidade a pagava antes do toque. Ancorar a troca no toque
  (`--anchor`) não resolveu; encurtar a janela para 15 frames levou o desvio mediano a −11 (D do GT)
  e −3 (D do Nuno). O que sobra deve vir do raio de 3 m do portão, que não dá para apertar sem
  bloquear as trocas de 11351 e 15766 (recebedor a 2–3 m da bola).
- **11351.** A cabeçada do Barça em 11342 não é detectada, e o resvalo de 11349 vai para o Athletic.
  Nem o `--aerial` resolve: no tracking, os jogadores do Barça ficam a 5–9 m da bola e do raio dela.
- **A bola morta é aceita como está** (seção 2.3): um D errado do detector passa direto para a
  submissão.

### 1.11 O que aprendemos sobre o GT (5 minutos públicos)

- **O toque do novo time acompanha as trocas.** Com o critério permissivo, as 12 trocas do GT têm um
  evento da bola com um jogador do novo time a até 3 m, a até 1 s da troca.
- **Mas o GT ignora muitos toques do adversário**, e nem o ângulo nem a força do toque separam os que
  contam dos que não contam (`diagnostico_toques.py`): com ângulo ≥ 45°, por exemplo, ficam 4 das 12
  trocas e 16 dos 34 toques ignorados.
- **Boa parte dos toques ignorados é erro de dado**, não regra do GT: bola alta projetada no lugar
  errado (12757), velocidades impossíveis de 50–64 m/s, e o goleiro com o time trocado (15652).

### 1.12 Validação no simulador

Sem acesso aos dados do jogo durante o desenvolvimento, o código foi testado num jogo simulado no
formato real dos arquivos: conduções, passes a ~15 m/s, 22% de interceptações (o GT troca no toque do
interceptador), bola ausente em rajadas, ruído de 0,3 m e 10% das posições da bola erradas (pulos para
outro objeto por 1 a 4 frames). O simulador é mais limpo que o jogo real (o baseline oficial faz 0,24
no jogo real e ~0,6 no simulador), então os números absolutos não valem para o jogo.

- **Filtro de pulos**: sem ele, o spellF1 simulado cai de 0,78–0,90 para 0,48–0,78, e a queda cresce
  com a fração de posições erradas. Com interpolação, sem filtro e toque 2,5, o preditor perdia até do
  baseline (0,54 contra 0,61).
- **Limiar adaptativo**: com limiar fixo de 4 m/s, o ruído gerava toques falsos (0,82); adaptativo, 0,94.
- **Sensibilidade**: variando cada parâmetro pela metade e pelo dobro, com ruído de 0,15 a 0,5 m, o
  spellF1 mudou no máximo 0,12; o mais sensível é o peso da proximidade.
- **Portão e goleiro**: no simulador o GT troca em toda interceptação e não há goleiros, então essas
  duas regras não têm o que corrigir ali, e só aparece o custo delas (o portão deixa o preditor mais
  conservador: 0,86 contra 0,90). Foram validadas por testes unitários e no GT público.
- **Toques aéreos**: num simulador com câmera (a câmera fica fora do campo, a 25 m de altura), a
  guarda contra pulos zerou os toques aéreos falsos (seção 1.6).

### 1.13 Próximos passos

1. **Janelas de validação.** Rotular 3 ou 4 blocos de 2–3 minutos fora da janela pública, um deles no
   2º tempo, com o painel **Posse por bloco** do viewer (`rotulos_blocos.csv`, convertido em gabarito
   por `analise/rotular_paradas.gt_blocos`). Isso diz se o 0,571 se sustenta fora dos 5 minutos em que
   tudo foi ajustado, desempata o peso do toque (0,5 contra 0,75) e explica a diferença de ritmo entre
   a janela pública e o jogo inteiro (seção 1.9).
2. **Posses curtas com evidência específica.** Em vez de afrouxar o critério para todas as trocas,
   liberar a posse curta só quando houver dois toques seguidos do mesmo time (passe completado ou
   condução). Um desvio ou um toque que o GT ignora costuma ser um toque isolado. Precisa ser testado:
   ideias parecidas (seção 1.6) pioraram.
3. **Fusão suave com a bola morta.** Pôr D como terceiro estado do Viterbi, com a probabilidade do
   detector como custo, para os toques poderem empurrar as bordas das paradas. Precisa que o detector
   exporte a probabilidade por frame, e não só 0/1.

---

## 2. Jogo inteiro: combinação com a bola morta do Nuno

### 2.1 Gerar o `paradas_D.csv`

Uma vez por clone (ou quando os dados mudarem). Rode **um comando por vez** e espere o prompt voltar
antes do próximo:

```powershell
python -m pip install scikit-learn seaborn matplotlib jupyter imageio-ffmpeg
python viewer/preprocess.py
python viewer/features.py
python analise/features_coletivas.py
cmd /c "python analise/audio_apito.py < NUL"
python -m jupyter nbconvert --to notebook --execute --output-dir analise/out --ExecutePreprocessor.timeout=-1 analise/02_arvore_bola_morta.ipynb
python -m jupyter nbconvert --to notebook --execute --output-dir analise/out --ExecutePreprocessor.timeout=-1 analise/03_detector_paradas.ipynb
```

- **`python -m pip` e `python -m jupyter`**: instalam e rodam no mesmo Python dos scripts, sem
  depender do PATH do Windows.
- **`< NUL` no áudio**: o `audio_apito.py` chama o ffmpeg sem desligar o teclado, então o que for
  digitado ou colado no terminal durante a extração vai para o ffmpeg como comando (a tecla `c` abre
  o prompt `Enter command:`; um `q` encerraria a extração no meio, sem erro). O `< NUL` faz o ffmpeg
  ignorar o teclado sem mexer no código.
- **`--output-dir analise/out`**: a cópia executada dos notebooks, com gráficos feitos a partir dos
  dados, vai para uma pasta fora do git, e os notebooks versionados ficam intactos.

Saídas usadas aqui: `analise/out/paradas_D.csv` (`frame,morta`, 1 = bola morta) e
`viewer/data/eventos.json` (as paradas e os apitos na timeline do viewer). O notebook 03 também grava
`evaluation/results/paradas_smooth50.csv` e `paradas_smooth125.csv` (as paradas com o time do
baseline), que servem de comparação.

### 2.2 Submissão e avaliação

```powershell
python posse/predictor.py --dead analise/out/paradas_D.csv --ball tracking_bola/out/viterbi_bola.csv --ball-units px --no-despike --out evaluation/results/posse_v2.csv
python evaluation/validate_submission.py evaluation/results/posse_v2.csv
python evaluation/run_evaluation.py
```

O `run_evaluation.py` pontua no GT público todas as submissões de `evaluation/results/`. Resultados na
seção 1.9. Para ver no viewer: `python viewer/serve.py`, **🎯 Carregar predição** com
`evaluation/results/posse_v1.csv` e, opcionalmente, **➕ Adicionar tracking** com
`posse/out/toques.csv`; o zoom da timeline em 1º tempo, 2º tempo ou vídeo inteiro mostra a predição
fora da janela do gabarito.

### 2.3 Critério de combinação

Não há fusão de probabilidades, e sim uma hierarquia: o detector do Nuno decide se a bola está viva
ou morta, e este preditor decide só de quem é a bola dentro do que ele marcou como vivo.

1. **Bola morta.** Dentro dos tempos de jogo, todo frame com `morta = 1` vira D. Fora dos tempos, X.
2. **Trechos vivos.** Cada sequência contínua de `morta = 0` é um trecho vivo independente; entre um
   trecho e outro, nada é carregado.
3. **Dono no início de cada trecho** (`restart_owner.decide`, seção 3.2): os votos de proximidade de
   50 frames antes a 5 depois do reinício; se a confiança for baixa, o Smart Tagging; se não houver
   nenhum dos dois, o tracking fraco.
4. **Dentro do trecho**, o Viterbi H/A da seção 1.

Consequências: um D que o detector marca em jogo corrido passa direto para a submissão e, pior,
parte o trecho vivo em dois, e o preditor decide um "reinício" onde não houve reinício. Uma parada que
o detector perde junta dois trechos, e a troca de posse da parada (um lateral para o outro time, por
exemplo) tem que ser achada pelo Viterbi como uma troca comum. O preditor nunca contesta o D. A queda
de 0,686 para 0,571 (seção 1.9) é o preço dessa hierarquia com fronteiras imperfeitas; a fusão suave
é o passo 3 da seção 1.13.

---

## 3. restart_owner.py

### 3.1 Trechos vivos

Lidos do arquivo passado em `--dead`, um rótulo por frame (199.319 linhas), em um de dois formatos:

- **`frame,morta`** (o `analise/out/paradas_D.csv` do detector de paradas): `morta = 1` é bola morta.
- **`frame,label`** (formato de submissão, como o GT público): `D` é bola morta e qualquer outro
  rótulo é vivo. Com `--live-ha`, só `H` e `A` são vivos; `D` e `X` são mortos. Serve para usar o GT
  público como `--dead` e testar a lógica nos 5 minutos públicos.

Dentro de cada metade avaliada, frames vivos consecutivos formam um trecho. O primeiro frame do
trecho é o **reinício**.

### 3.2 Dono no reinício

Durante a bola morta, o cobrador vai até a bola parada e fica ao lado dela. Quem está colado na bola
logo antes do reinício é quem reinicia.

Para o reinício no frame *s*:

1. **Janela:** de *s* − 50 a *s* + 5 (2 s antes, 0,2 s depois). Depois disso a bola já saiu do
   cobrador.
2. **Bola:** a do CSV em `--ball` ou, sem ele, a do ATD. Onde ela some, repete a última posição
   vista nos 5 s anteriores (bola parada não se move).
3. **Voto por frame:** distância, em metros, da bola ao jogador mais perto de cada time. O time mais
   perto vota se o jogador dele estiver a até 5 m e pelo menos 1 m mais perto que o adversário mais
   próximo. Bola disputada ou longe de todos não vota.
4. **Sem nenhum voto:** tenta de novo com os 2 s depois do reinício.
5. **Decisão** (coluna `source`):

| source | condição | dono |
|---|---|---|
| `tracking` | ≥ 5 votos e confiança ≥ 0,5 | time com mais votos |
| `tagging` | tracking fraco ou mudo, e há `SET PIECES` com time cobrindo o reinício | time da tag |
| `tracking_fraco` | tracking fraco e sem tag | time com mais votos, pouco confiável |
| `nenhum` | menos de 5 votos e sem tag | `X` |

Confiança = |votos_H − votos_A| / (votos_H + votos_A).

**Smart Tagging:** só `SET PIECES` com `Team` igual a Barcelona ou Atletic Club, com 3 s de folga em
volta do intervalo (a tag é uma janela em torno da cobrança, não o instante). Ficam de fora os
laterais (`Team = N/A`) e o `Kick Off (After Goal)`, em que não se sabe se `Team` é quem sai jogando
ou quem marcou. O tracking tem prioridade quando é claro.

Quando chamado pelo `predictor.py`, o reinício usa a bola já filtrada e os times com a correção dos
goleiros (seção 1.3), o que importa sobretudo nos tiros de meta.

Todos os números (2 s, 5 m, 1 m, 5 votos, 0,5, 3 s) estão na seção `# Params` do script. São
escolhas com justificativa de futebol, não ajustadas no GT público.

### 3.3 Como rodar

```powershell
python posse/restart_owner.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball caminho\da\bola.csv
python posse/restart_owner.py --dead analise/out/paradas_D.csv --ball caminho\da\bola.csv --write-submission evaluation/results/reinicio_constante.csv
```

| argumento | o que faz |
|---|---|
| `--dead` | arquivo de bola morta (obrigatório): `frame,morta` ou `frame,label` |
| `--live-ha` | só H e A contam como vivo |
| `--ball` | CSV da bola; sem ele, usa a bola do ATD |
| `--ball-cols` | colunas de frame, x e y (padrão `frame,x,y`) |
| `--ball-units` | `norm` (0 a 1, como o ATD) ou `m` (metros) |
| `--ball-frame-base` | `0` (frame do vídeo) ou `1` (frame_id do ATD) |
| `--no-tagging` | ignora o Smart Tagging |
| `--write-submission` | escreve a submissão de trecho constante: D na bola morta e o dono do reinício no trecho vivo inteiro (a linha de base "nenhuma troca de posse") |
| `--gt`, `--out` | caminhos do GT e da tabela de saída; os padrões servem |

Saídas: o resumo no terminal (origem das decisões, concordância entre tracking e tagging e, na janela
pública, cada reinício com o dono previsto e o do GT) e `posse/out/restart_owner.csv`, uma linha por
reinício.

### 3.4 Resultado na janela pública

5 reinícios, fronteiras do GT (`--live-ha`):

| bola | acertos | erros |
|---|---|---|
| ATD | 3 | saída inicial sem decisão; 12989 decidido errado com confiança baixa |
| fusão ATD + Roboflow | 5 | — |

Amostra pequena: confirma que a lógica funciona, não mede a taxa de acerto no jogo inteiro.

---

## 4. diagnostico_toques.py

Nos 5 minutos públicos (trechos vivos do próprio GT), acha os eventos da bola com um limiar baixo e
separa dois grupos:

- **TROCA:** para cada troca do GT, o evento a até 1 s em que o novo time está mais perto da bola. Se
  não houver, lista todos os eventos da janela, com o time e as distâncias.
- **IGNORADO:** eventos com o time sem a posse mais perto da bola, a mais de 1 s de qualquer troca do
  GT: os toques do adversário que o GT não contou.

Para cada evento: ângulo da mudança de direção (só com a bola a ≥ 3 m/s dos dois lados), mudança do
vetor velocidade, velocidades antes e depois e distâncias da bola a cada time. No fim, quantos de cada
grupo ficam acima de cada limiar de ângulo e de velocidade.

```powershell
python posse/diagnostico_toques.py --ball tracking_bola/out/viterbi_bola.csv --ball-units px --no-despike
```

Aceita os mesmos argumentos de bola do preditor, além de `--no-despike`, `--no-gk-fix` e
`--min-speed`.

---

## 5. comparar_bolas.py

Roda o preditor com as três bolas, cada uma com e sem filtro de pulos, num processo só (o ATD é lido
uma vez), nos 5 minutos públicos com as fronteiras do GT e as demais melhorias ligadas:

| bola | arquivo | unidades |
|---|---|---|
| fusão | `data/tactical/BAR-ATH_tactical_ball_tracker.csv` | `norm` |
| Viterbi | `tracking_bola/out/viterbi_bola.csv` | `px` |
| Kalman | `tracking_bola/out/kalman_bola.csv` | `px` |

```powershell
python posse/comparar_bolas.py
python posse/comparar_bolas.py --extra --touch-weight 1.0
```

`--extra` repassa opções ao preditor em todas as rodadas (vai por último). Bolas cujo arquivo não
existe são puladas. Cada rodada grava `posse/out/comparacao_<bola>_<filtro>.csv`, e a tabela final
traz spellF1, macroF1, posses previstas, trocas certas, perdidas, inventadas, bloqueadas pelo portão e
a fração de posições descartadas pelo filtro.

---

## 6. varrer_portao.py

Roda o preditor com várias janelas do portão, num processo só (o ATD é lido uma vez), com a bola
campeã (Viterbi do tracking, sem filtro de pulos):

- **simétrica**: o evento libera a troca de N frames antes a N depois, com N = 25 (o padrão
  anterior), 20, 15 (o padrão atual), 10 e 5;
- **só antes**: 25 frames depois e só a parte de antes encolhe, para 15, 10, 5 e 0. É a variante
  que mira as trocas adiantadas (seção 1.10): com pouca janela antes do evento, a troca não pode cair
  muito antes dele.

```powershell
python posse/varrer_portao.py
python posse/varrer_portao.py --dead nuno
python posse/varrer_portao.py --janelas 25 30 40 50 75 100 --sem-portao
python posse/varrer_portao.py --extra --touch-weight 0.75
```

`--janelas` troca a lista por janelas simétricas escolhidas (e não roda a família "só antes");
`--sem-portao` acrescenta duas rodadas sem portão, com o peso do toque de 1,5 (a política antiga) e
de 0,5 (só o portão muda).

**Resultados no GT público** (bola campeã, peso do toque 0,5; com o portão, certas, perdidas e
inventadas são iguais em todas as linhas: 4/8/2 com o D do GT e 3/9/3 com o do Nuno):

| antes / depois | spellF1, D do GT | macroF1, D do GT | spellF1, D do Nuno | macroF1, D do Nuno | desvio das certas (GT / Nuno) | bloqueadas |
|---|---|---|---|---|---|---|
| 25 / 25 (anterior) | 0,6857 | 0,9604 | 0,5714 | 0,9191 | −20 / −19 | 0 |
| 20 / 20 | 0,6857 | 0,9603 | 0,5714 | 0,9190 | −20 / −19 | 0 |
| **15 / 15** | **0,6857** | **0,9617** | **0,5857** | **0,9204** | **−11 / −3** | 0 |
| 10 / 10 | 0,6857 | 0,9617 | 0,5857 | 0,9204 | −11 / −3 | 1 |
| 5 / 5 | 0,6857 | 0,9616 | 0,5857 | 0,9207 | −12 / −4 | 2 |
| 0 / 25 | 0,6857 | 0,9609 | 0,5857 | 0,9206 | −19 / −17 | 1 |
| 30 / 30 a 100 / 100 | 0,6857 | 0,9604 | 0,5714 | 0,9191 | −20 / −19 | 0 |

Sem portão, para comparar (as trocas mudam, então a linha traz certas/perdidas/inventadas):

| | D do GT: spellF1, macroF1, posses, certas/perdidas/inventadas | D do Nuno: spellF1, macroF1, posses, certas/perdidas/inventadas |
|---|---|---|
| sem portão, peso 1,5 | 0,5684, 0,9481, 21, 7/5/9 | 0,5405, 0,9074, 20, 7/5/8 |
| sem portão, peso 0,5 | 0,6207, 0,9496, 12, 4/8/3 | 0,5571, 0,9157, 11, 3/9/3 |

- De 20 a 100 frames, nada muda: com a janela larga, quem decide é a suficiência.
- A janela não muda quais trocas acontecem, só quando. Com 15 frames depois do evento, as trocas
  deixam de sair ~20 frames adiantadas; o salto acontece entre 20 e 15, então os eventos "cedo"
  ficam 15–20 frames antes da troca do GT.
- Encolher só a parte de antes quase não muda o desvio: a troca não caía antes do evento que a
  liberava, e sim depois de um evento que já era cedo.
- Abaixo de 15, o portão começa a bloquear trocas verdadeiras (por enquanto, só posses curtas que já
  estavam perdidas).
- +0,014 de spellF1 com o D do Nuno é uma posse passando de um limiar de sobreposição a mais: dentro
  do ruído, mas na mesma direção nos dois cenários e sem custo.

Sem `--dead`, usa as fronteiras de bola morta do GT (`--live-ha`); com `--dead nuno`, o
`analise/out/paradas_D.csv`. `--extra` repassa opções ao preditor em todas as rodadas. Cada rodada
grava `posse/out/portao_<antes>_<depois>.csv`. A tabela traz spellF1, macroF1, posses, trocas certas,
perdidas, inventadas, bloqueadas pelo portão, o desvio mediano das trocas certas (negativo =
adiantada) e a fração dos frames vivos em que a troca fica liberada para cada time.