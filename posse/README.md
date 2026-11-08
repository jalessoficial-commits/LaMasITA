# posse/

Rotulagem de posse (H ou A) dentro dos trechos de bola viva. A bola morta (D) vem do detector de
outro membro da equipe; esta pasta só decide **de quem é a bola** quando ela está viva.

| script | o que faz |
|---|---|
| `restart_owner.py` | o dono no reinício: todo reinício tem um dono claro (CHALLENGE.md), então cada trecho vivo começa com um time conhecido |
| `predictor.py` | o preditor completo: parte do dono do reinício, decide quando a posse troca dentro do trecho e gera a submissão |
| `diagnostico_toques.py` | compara os eventos da bola perto das trocas do GT público com os toques do adversário que o GT ignora |

Nenhum script envia dados para fora da máquina. Tudo o que é derivado dos dados do jogo (tabelas por
frame, camadas para o viewer) fica em `posse/out/`, que está no `.gitignore`.

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

Duas decisões vêm do `tracking_bola/README.md`:

- **Sem interpolação.** O Kalman e o Viterbi do passo 15 não filtraram erros, e a decisão foi deixar a
  camada de posse atravessar as lacunas da bola (seção 6). Além disso, uma reta entre duas medidas
  ruidosas cria quebras de velocidade nas pontas, que virariam toques falsos.
- **Filtro de pulos** (desligável com `--no-despike`). O modelo final do tracking acerta 67,1% dos
  frames com bola no teste, e cerca de 1 em 10 posições que ele dá está errada, em geral um pulo para
  outro objeto por poucos frames. Cada pulo cria uma quebra de velocidade, ou seja, um toque falso.
  Cada posição é comparada com a mediana, coordenada a coordenada, das posições dos 5 frames antes e
  5 depois; se estiver a mais de 3 m, é descartada. A mediana acompanha a bola em linha reta e na
  quebra de direção de um toque real; um pulo de até 5 frames não a arrasta.

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
frames cada). A regra atual só mexe nos dois slots de goleiro. Limitação: se a identidade do goleiro
passar de forma duradoura para um terceiro slot, esse trecho não é corrigido; pelo diagnóstico, nenhum
outro slot passa de 7% como mais recuado, e esse slot (42) tem perfil de zagueiro (9% do tempo na área,
43 m de mediana até a linha de fundo).

O terminal imprime os slots escolhidos e em quantos frames o time de cada um foi corrigido.

### 1.4 Evidência por frame

| fonte | o que é | peso |
|---|---|---|
| **Toque estrito** | pico local da mudança do vetor velocidade da bola (média dos 4 frames antes contra a dos 4 depois, pelo menos 3 velocidades válidas de cada lado), acima de 3× o ruído típico da bola (mínimo 4 m/s), fora de voo (seção 1.5); vai para o time do jogador mais perto, se ele estiver a até 2 m e 0,75 m mais perto que o adversário | 0,5 com o portão (1,5 sem) |
| **Proximidade** | a cada frame com bola, o time mais perto, pesado pela distância (some acima de ~3 m) e pela folga sobre o adversário | até 0,08 por frame |
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

A troca **para o time X** só é permitida a até ±1 s (25 frames) de um evento que a libere
(desligável com `--no-gate`, que volta à política anterior, com toque de peso 1,5):

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

### 1.6 Como rodar

Teste nos 5 minutos públicos, com as fronteiras de bola morta do GT (`--live-ha`):

```powershell
python posse/predictor.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball data\tactical\BAR-ATH_tactical_ball_tracker.csv --out posse/out/teste_fusao.csv
python posse/predictor.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball tracking_bola/out/viterbi_bola.csv --ball-units px --no-despike --out posse/out/teste_viterbi.csv
```

Jogo inteiro, com o arquivo de bola morta:

```powershell
python posse/predictor.py --dead caminho\do\bola_morta.csv --ball caminho\da\bola.csv --out evaluation/results/<time>.csv
python evaluation/validate_submission.py evaluation/results/<time>.csv
```

| argumento | o que faz |
|---|---|
| `--dead` | arquivo de bola morta (obrigatório) |
| `--live-ha` | só H e A contam como vivo (para usar o GT público como `--dead`) |
| `--ball`, `--ball-cols`, `--ball-units`, `--ball-frame-base` | a bola e o seu formato (seção 1.2) |
| `--no-despike` | desliga o filtro de pulos |
| `--no-gk-fix` | desliga a correção dos goleiros |
| `--no-gate` | desliga o portão (política anterior, toque com peso 1,5) |
| `--touch-weight` | peso do toque estrito (padrão 0,5 com o portão, 1,5 sem) |
| `--tag-weight` | peso das fases do Smart Tagging (padrão 0) |
| `--out` | arquivo da submissão; é sobrescrito a cada execução |

O ATD é lido inteiro (~1–2 min).

### 1.7 Saídas

- **Submissão**: D na bola morta, H/A nos trechos vivos, X fora das metades.
- **`posse/out/toques.csv`**: camada para o viewer (**➕ Adicionar tracking**) com a bola em cada toque
  estrito, na cor do time. Junto com a submissão em **🎯 Carregar predição**, mostra por que o preditor
  trocou (ou não) em cada lance. É sobrescrito a cada execução, então corresponde sempre à última.
- **Terminal, com o GT público**: spellF1 e macroF1 do avaliador oficial; cada troca do GT com a troca
  prevista correspondente (até 2 s, no mesmo sentido), o desvio em frames e se estava liberada; e as
  trocas previstas sem par no GT.

### 1.8 Resultados no GT público

Todas as rodadas abaixo usam as fronteiras de bola morta do próprio GT (`--live-ha`), o que infla os
números, sobretudo o macroF1 (todo D sai certo): **não se comparam com os baselines oficiais**. São 17
posses e 12 trocas de posse com a bola viva; uma posse a mais ou a menos move o spellF1 em ~6 pontos.
As duas últimas linhas acrescentam cada mudança à configuração da linha anterior (bola do Viterbi,
filtro desligado).

| configuração | spellF1 | macroF1 | posses (GT 17) | trocas inventadas |
|---|---|---|---|---|
| fusão ATD + Roboflow, filtro ligado, toque 1,5 | 0,533 | 0,939 | 22 | 10 |
| fusão, filtro desligado | 0,435 | 0,902 | 29 | 17 |
| Viterbi do tracking, filtro ligado | 0,544 | 0,934 | 22 | 10 |
| Viterbi do tracking, filtro desligado | 0,550 | 0,943 | 23 | 11 |
| + goleiro (versão "mais recuado da área") | 0,568 | 0,948 | 21 | 9 |
| + portão (toque necessário, peso 0,5) | **0,686** | **0,960** | 11 | 2 |

- **O filtro de pulos é necessário para a fusão** (−10 pontos sem ele) e indiferente para o Viterbi,
  que já impede saltos impossíveis. Fusão com filtro e Viterbi empatam dentro do ruído.
- **O Kalman do tracking** ficou abaixo dos dois.
- **A correção do goleiro** tirou exatamente o par falso 15647/15694, sem mudar mais nada.
- **O portão** cortou as trocas inventadas de 9 para 2, sem bloquear nenhuma troca verdadeira (as 12
  ficaram liberadas). Em troca, o preditor ficou conservador: 6 trocas previstas, 4 certas, e perdeu
  todas as posses curtas (11553→11624, 15346→15365, 15766→15799). Elas estavam liberadas; quem barrou
  foi a suficiência (toque de peso 0,5 mais a proximidade). O portão também é permissivo: a troca fica
  liberada em ~73% dos frames vivos para H e ~56% para A.
- A regra atual do goleiro (slots de goleiro com o time pelo lado do campo) ainda não foi medida no
  jogo real.

### 1.9 O que aprendemos sobre o GT (5 minutos públicos)

- **O toque do novo time acompanha as trocas.** Com o critério permissivo, as 12 trocas do GT têm um
  evento da bola com um jogador do novo time a até 3 m, a até 1 s da troca.
- **Mas o GT ignora muitos toques do adversário**, e nem o ângulo nem a força do toque separam os que
  contam dos que não contam (`diagnostico_toques.py`): com ângulo ≥ 45°, por exemplo, ficam 4 das 12
  trocas e 16 dos 34 toques ignorados.
- **Boa parte dos toques ignorados é erro de dado**, não regra do GT: bola alta projetada no lugar
  errado (12757), velocidades impossíveis de 50–64 m/s, e o goleiro com o time trocado (15652).

### 1.10 Validação no simulador

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

### 1.11 Próximos passos

1. Medir a regra nova do goleiro no GT público.
2. Rever a suficiência: com o portão, o preditor perde as posses curtas. Variar o peso do toque
   (`--touch-weight`) mostra quantas trocas falsas voltam para cada troca perdida recuperada; com 12
   trocas, isso indica a forma da troca, não um valor exato.
3. Rodar no jogo inteiro com o arquivo de bola morta e validar a submissão.

---

## 2. restart_owner.py

### 2.1 Trechos vivos

Lidos do arquivo passado em `--dead`, um rótulo por frame (formato de submissão, 199.319 linhas).

- Padrão: `D` é bola morta e qualquer outro rótulo é vivo.
- Com `--live-ha`: só `H` e `A` são vivos; `D` e `X` são mortos. Serve para usar o GT público como
  `--dead` e testar a lógica nos 5 minutos públicos.

Dentro de cada metade avaliada, frames vivos consecutivos formam um trecho. O primeiro frame do
trecho é o **reinício**.

### 2.2 Dono no reinício

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

### 2.3 Como rodar

```powershell
python posse/restart_owner.py --dead evaluation/gt/BAR-ATH_possession_gt_public.csv --live-ha --ball caminho\da\bola.csv
python posse/restart_owner.py --dead caminho\do\bola_morta.csv --ball caminho\da\bola.csv --write-submission evaluation/results/reinicio_constante.csv
```

| argumento | o que faz |
|---|---|
| `--dead` | arquivo de bola morta (obrigatório) |
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

### 2.4 Resultado na janela pública

5 reinícios, fronteiras do GT (`--live-ha`):

| bola | acertos | erros |
|---|---|---|
| ATD | 3 | saída inicial sem decisão; 12989 decidido errado com confiança baixa |
| fusão ATD + Roboflow | 5 | — |

Amostra pequena: confirma que a lógica funciona, não mede a taxa de acerto no jogo inteiro.

---

## 3. diagnostico_toques.py

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

Aceita os mesmos argumentos de bola do preditor, além de `--no-despike` e `--no-gk-fix`.