# Rastreamento da bola — documento técnico (parte 1: detecção da bola)

> **Escopo:** este documento cobre o rastreamento da bola a partir do vídeo `tactical`
> (Barcelona x Athletic Club).

## 1. Resumo

O ATD fornecido (rastreamento automático da Metrica) **não tem bola em ~41% dos frames avaliados** e,
onde tem, a posição erra por dezenas de pixels. Construímos um rastreador de bola próprio a partir do
vídeo, sem treinar nenhum modelo (só pesos públicos). Os parâmetros foram escolhidos à mão, comparando
as configurações no viewer; frames anotados à mão serviram **só para medir** o desempenho do que
implementamos, sem treinar nem escolher nada, e nunca como saída do sistema.

**Modelo final:** fusão do ATD com o detector de bola do Roboflow (YOLOv8 treinado em futebol), com os
candidatos filtrados por uma **máscara do campo de 2 m** e a regra de fusão com **`--solo-conf 0,8`** e
**`--override-conf 0,8`**. Esse modelo foi então rodado nos **158.857 frames avaliados** do jogo: a bola
passou a existir em **72,8%** deles, contra 59,3% do ATD, e a posição foi exportada em coordenadas do campo,
no formato do ATD, como substituta da bola dele (passos 13 e 14).

> **Qual bola a submissão usa.** A camada de posse (`posse_ml/`) usa o **rastro por Viterbi** do passo 15
> (`out/viterbi_bola.csv`), calculado sobre os candidatos deste mesmo detector com a máscara de 2 m, e não a
> fusão. Este documento conclui, só pelas métricas de detecção, que fusão e Viterbi são pontos da mesma troca
> entre acerto e falso alarme; a escolha do Viterbi veio depois, medida na posse (`posse_ml/README.md`,
> seções 5.2, 5.3 e 5.5). A fusão continua sendo usada pelo detector de bola morta (`analise/`).

**Resultado no conjunto de teste** (93 frames novos, não usados para escolher nada; tolerância de 40 px):

| | ATD | Modelo final |
|---|---|---|
| Acerto (79 frames com bola; margem ±11) | 53,2% | **67,1%** |
| Acerto até 10 px | 30,4% | **65,8%** |
| Erro mediano nos acertos | 8,2 px | **0,4 px** |
| Acerto onde o ATD não tem bola (29 frames) | 0% | **24,1%** |
| Acerto onde o ATD tem bola (50 frames) | 84,0% | **92,0%** |
| Falso alarme (14 frames sem bola) | 7,1% | 7,1% |
| F1 | 64,6% | **76,8%** |

A tabela abaixo traz a métrica de cada etapa nos 292 frames anotados (só medição).

| Etapa (292 frames anotados, tolerância de 40 px) | Acerto | Onde o ATD não tem bola | Falso alarme | F1 |
|---|---|---|---|---|
| ATD sozinho | 48,6% | 0% | 15,2% | 59,3% |
| YOLO do COCO sem treino | não superou o ATD — descartado | | | |
| FootAndBall sem treino | ~1% — descartado | | | |
| Fusão ATD + WASB-SBDT | 54,4% | 15,3% | 24,2% | 60,6% |
| Fusão ATD + Roboflow, override 0,8 | 65,3% | 31,6% | 48,5% | 69,1% |
| **Fusão ATD + Roboflow, override 0,8, máscara do campo** | **66,4%** | **31,6%** | **24,2%** | **73,3%** |

A última linha é a máscara de 3 m; a de 2 m foi a adotada no modelo final.
Tabela completa na seção 4.

## 2. Dados e modelos usados

| Ativo | Uso | Observação |
|---|---|---|
| Vídeo `tactical` (1920×1080, 25 fps, 199.319 frames) | entrada dos detectores | não redistribuído; `data/` fica fora do Git |
| ATD bruto (bola em coordenadas do campo, 0–1) | linha de base e parte da fusão | bola ausente = `NaN` |
| Homografia por frame (válida em 100% dos frames do vídeo `tactical`) | projeta a bola do ATD em pixels e leva candidatos de volta ao campo (máscara) | `starter/load_homography.py` |
| Dimensões do campo (`games.py`, 105,3 × 68 m) | converter a margem da máscara de metros para o campo normalizado | |
| Smart Tagging | só diagnóstico (`gap_map.py`: relação entre lacunas de bola e bolas paradas) | não entra no rastreador |
| Blocos avaliados (`games.half_video_blocks`) | 1º tempo 11.185–90.651; 2º tempo 110.808–190.197 (158.857 frames) | só esses frames são processados |
| 292 frames anotados à mão (`annotate_ball.py`) | só **medição**: métrica de cada etapa | 259 com bola visível, 33 sem bola; não usados para escolher nada |
| 93 frames anotados à mão, em arquivo separado | só **medição**: métrica do modelo final | frames 301–400 da mesma lista sorteada (79 com bola, 14 sem); não usados para escolher nada |

Modelos pré-treinados (todos públicos; o commit de cada repositório clonado fica em
`tracking_bola/third_party/*_commit.txt`):

| Modelo | Origem | Licença | Pesos | Papel |
|---|---|---|---|---|
| **Detector de bola do Roboflow** (YOLOv8, 1 classe: `ball`) | github.com/roboflow/sports | código MIT; inferência via `ultralytics` (AGPL-3.0) | `football-ball-detection.pt`, link do Google Drive em `examples/soccer/setup.sh` | **modelo final** |
| WASB-SBDT | github.com/nttcom/WASB-SBDT (Tarashima et al., BMVC 2023) | MIT | `wasb_soccer_best.pth.tar`, link no README deles | comparado, não escolhido |
| YOLOv8m (COCO, `sports ball`) | `ultralytics` | AGPL-3.0 | `yolov8m.pt` | descartado |
| FootAndBall | github.com/jac99/FootAndBall (Komorowski et al., VISAPP 2020) | MIT | `models/model_20201019_1416_final.pth` | descartado |

**Conformidade.** As anotações à mão servem somente para medir o desempenho; não treinam nem escolhem
nada, e nada do que o sistema produz foi rotulado à mão. Os scripts processam o vídeo apenas na máquina local e não enviam dados a nenhum
serviço externo (só baixam pesos públicos). Os arquivos derivados do vídeo (CSVs de posição por frame e
recortes de imagem em `out/`) não são publicados: ficam no `.gitignore`, e este documento traz apenas
métricas agregadas.

## 3. Método, passo a passo

Cada passo é um script em `tracking_bola/`; os comandos estão na seção 7.

**Passo 0 — Entender a falha do ATD (`gap_map.py`, `check_projection.py`).**
O ATD tem bola em 59,3% dos frames avaliados (94.214 de 158.857, conforme o DATA.md). O `gap_map.py`
distribui as lacunas por duração e as cruza com as bolas paradas do Smart Tagging; o `check_projection.py`
projeta a bola do ATD no vídeo com a homografia, para ver se cai sobre a bola (a homografia vale para o
plano do chão, então isso importa sobretudo para bola no ar).

**Passo 1 — Frames anotados para medição (`annotate_ball.py`).**
Os dados fornecidos não trazem a posição verdadeira da bola. Anotamos à mão a bola em frames sorteados dos
dois tempos, no máximo um por bloco de 125 frames (5 s). Cada frame recebe `ball`, `not_visible`,
`uncertain` ou `skip`. Resultado: 292 frames, 259 com bola visível e 33 sem. Servem só para medir cada
etapa; as escolhas foram feitas à mão, olhando as saídas no viewer.

**Passo 2 — Medir o ATD (`eval_atd.py`, `check_atd_nature.py`).**
A bola do ATD é projetada em pixels e comparada com o clique. Verificamos também se as posições do ATD
são repetidas ou interpoladas: só 0,1% repetem o frame vizinho e 10,7% têm movimento exatamente linear;
os 89,2% restantes parecem medidas novas, com erro mediano de 15,7 px. O erro do ATD vem das próprias
detecções.

**Passo 3 — Tolerância de acerto (`measure_ball.py`).**
A bola mede **8,2 px** de diâmetro. As comparações finais usam **40 px** (~5 diâmetros), a regra do
script; `--tol` muda o valor. A seção 4 traz também o acerto a 10 px.

**Passo 4 — YOLO do COCO sem treino (`detect_yolo.py`) — descartado.**
Classe `sports ball`, em ladrilhos na resolução nativa (reduzir o frame inteiro apaga a bola). Não superou
o ATD: o COCO quase não tem bolas pequenas em plano aberto.

**Passo 5 — WASB-SBDT sem treino (`detect_wasb.py`, `diagnose_wasb.py`).**
O WASB (HRNet de ~1,5 M de parâmetros, 3 frames consecutivos na entrada, mapa de calor na saída) era o
melhor publicado no benchmark de futebol do próprio artigo (ISSIA-CNR, F1 88,3). Reduzir o frame para a
entrada de 512×288 apaga a bola (2% de acerto); por isso o frame é cortado em ladrilhos de 512×288 com
20% de sobreposição, depois de reduzido por um fator. Ficamos com **0,6** (~9 ladrilhos por frame), escolhido
à mão; a métrica em 100 frames anotados foi 0,4 → 29%, 0,5 → 58%, 0,6–0,9 → 61–65%. Limiar do mapa 0,2, vários
candidatos por frame com a confiança de cada um. Resultado: muito preciso (2,3 px de erro mediano nos
acertos), mas com muitos erros longe e falso alarme alto em limiar baixo.

**Passo 6 — Fusão ATD + detector (`fuse_atd_wasb.py`).**
Regra por frame, válida para qualquer detector com CSV `frame,x,y,conf` (candidatos com conf < 0,2 são
descartados antes):

1. **ATD sem bola:** usa o melhor candidato do detector se a confiança for ≥ `--solo-conf`; senão, sem bola.
2. **ATD com bola:**
   - 2a. algum candidato a até 40 px do ATD (`--radius`) → vale o de maior confiança entre eles (os dois
     veem a mesma bola; a posição do detector é mais precisa);
   - 2b. nenhum perto, mas o melhor tem confiança ≥ `--override-conf` → vale o detector (ele passa por
     cima do ATD);
   - 2c. caso contrário → vale o ATD.

Com o WASB (`--solo-conf 0,5`, override desligado) a fusão chegou a 54,4% de acerto.

**Passo 7 — FootAndBall sem treino (`detect_footandball.py`, `diagnose_footandball.py`) — descartado.**
Rede totalmente convolucional que opera no frame inteiro. Acertou ~1%. O diagnóstico do mapa de confiança
em 60 frames mostrou que a rede quase não responde à bola na escala nativa (confiança mediana 0,001) e só
melhora ampliando o frame (33% dos frames com confiança ≥ 0,2 em escala 2,0, a ~4× o custo); a bola do
nosso vídeo é pequena para o que ela viu no treino. Pré-processamento conferido contra o código original.

**Passo 8 — Busca por alternativas.**
Revisamos a literatura recente. Os métodos que superam o WASB (TOTNet, TrackNetV3–V5) foram avaliados só
em esportes de raquete e não têm pesos de futebol; os trabalhos recentes de futebol usam detectores
genéricos ajustados à bola. Daí o candidato seguinte: o detector de bola do Roboflow, um YOLOv8 treinado
especificamente na bola de futebol (vídeos da Bundesliga, frames cortados em ladrilhos), com pesos públicos.

**Passo 9 — Detector de bola do Roboflow (`detect_yolo.py --weights football-ball-detection.pt --class-id 0`).**
Mesmo script do passo 4, com os pesos do Roboflow e ladrilhos de 640×640 (como no código deles). Olha um
frame por vez. Contra o WASB: acerta mais (57,9% contra 45,9% em limiar 0,2), recupera o dobro onde o ATD
não tem bola (39,8% contra 22,4%) e é ainda mais preciso nos acertos (0,7 px), mas dá falso alarme em
quase todo frame sem bola (97% em 0,2; 36% em 0,8). Sua confiança quase não passa de 0,85.

**Passo 10 — Fusão ATD + Roboflow com override.**
Com `--solo-conf 0,8` o falso alarme cai. Ligar `--override-conf 0,8` corrigiu erros do ATD onde ele tem
bola: o acerto nesses frames subiu de 78,3% para 85,7%, e o F1 da fusão foi a 69,1%.

**Passo 11 — Máscara do campo (`show_false_alarms.py`, `mask_pitch.py`).**
O `show_false_alarms.py` lista e recorta os frames sem bola em que a fusão deu uma posição. Olhando os
recortes, boa parte dos falsos alarmes eram **objetos fora do campo**
detectados como bola com confiança alta. Consistência temporal não resolveria, porque um objeto parado é
contínuo. O
`mask_pitch.py` leva cada candidato do pixel para o campo pela inversa da homografia do frame e descarta os
que caem a mais de uma margem das linhas; o próximo candidato, muitas vezes a bola de verdade, assume.
Com 3 m: falso alarme da fusão de 48,5% para 24,2%, **sem perder nenhum acerto** onde o ATD não tem bola,
p90 do erro de 683 para 73 px, F1 73,3%. A margem adotada foi **2 m**.
A margem existe porque a bola de verdade sai do campo em laterais e escanteios, e uma bola alta é
projetada mais longe do que está.

**Passo 12 — Segunda medição (`annotate_ball.py --start 300 --n 400`).**
Detector, limiares, override e margem foram escolhidos à mão, comparando as saídas no viewer. Para ter
uma segunda medida, anotamos frames novos (posições 301–400 da mesma lista sorteada,
diferentes dos 300 primeiros; 93 frames aproveitados, 79 com bola e 14 sem) num arquivo separado e medimos
neles, sem mudar nenhum parâmetro, o ATD e a fusão ATD + Roboflow, com e sem máscara. O modelo final é a fusão ATD + Roboflow, override 0,8, máscara de 2 m. No teste ele acerta 67,1%
contra 53,2% do ATD (diferença acima da margem de ±11 pontos), com o mesmo falso alarme do ATD (1 de 14
frames sem bola) e F1 de 76,8% contra 64,6%. Sem a máscara, a mesma fusão teria acerto igual, mas falso
alarme de 21,4% e erro p90 de 372 px (contra 5,8 px com a máscara): nos 93 frames, como nos 292, a máscara
só removeu candidatos errados. A comparação com a fusão ATD + WASB é da primeira medição (292 frames);
ela não foi rodada nos 93.

**Passo 13 — Execução no jogo todo (`run_champion_full.py`).**
O modelo final foi rodado nos 158.857 frames avaliados (os dois tempos): detector do Roboflow em
ladrilhos → máscara do campo de 2 m → fusão com o ATD, com os mesmos parâmetros. O vídeo é lido uma vez, em ordem; a detecção salva a cada 500 frames e retoma de onde parou se
for interrompida. Rodou na GPU do notebook do autor (NVIDIA RTX 3050 de 6 GB, PyTorch com CUDA) em cerca
de 27 h; na CPU a estimativa era de ~385 h. A saída (`out/full_bola.csv`) é a posição da bola em pixels de
cada frame avaliado com bola, com a regra da fusão que a produziu. Fica em `out/`, fora do Git.

| Frames avaliados | ATD com bola | Tracker com bola |
|---|---|---|
| 1º tempo (79.467) | 59,3% | 71,1% |
| 2º tempo (79.390) | 59,3% | 74,6% |
| **Total (158.857)** | **59,3%** | **72,8%** |

| Situação no frame avaliado | Frames | % |
|---|---|---|
| ATD e tracker com bola | 94.214 | 59,3% |
| Só o tracker (bola recuperada onde o ATD estava cego) | 21.475 | 13,5% |
| Só o ATD | 0 | 0,0% |
| Nenhum dos dois | 43.168 | 27,2% |

O tracker recuperou cerca de um terço dos 64.643 frames em que o ATD não tinha bola e não perdeu nenhum
frame em que o ATD tinha. Origem da posição nos frames em que o ATD tinha bola (94.214): **refinada** pelo
detector, que concordou com o ATD, em 66.454 (70,5%); **trocada** pelo detector confiante (override) em
12.847 (13,6%); **mantida** a do ATD em 14.913 (15,8%). Parte dos 27,2% sem bola é bola de fato fora de vista
(escondida por jogadores, fora do campo, reposições): nos frames anotados, 11% eram "bola não visível".

**Passo 14 — Bola em coordenadas do campo, substituindo a do ATD (`export_ball.py`).**
Converte a saída do passo 13 para coordenadas normalizadas do campo (0–1, as mesmas do ATD) pela inversa
da homografia de cada frame e grava `data/tactical/BAR-ATH_tactical_ball_tracker.csv`, ao lado do ATD, com
uma linha por frame do vídeo (0 a 199.318): `frame, x, y, fonte`. A coluna `fonte` diz de onde veio a posição
(`detector`, `detector_refinou`, `detector_trocou`, `atd`, `sem_bola`); onde a fusão manteve o ATD, o
valor original é copiado sem arredondamento. Fora dos tempos avaliados, que não contam para nada, vale a
bola original do ATD (`atd_fora_do_jogo`). Antes de gravar, o script imprime a comparação com o ATD
mostrada acima. O arquivo é dado derivado do vídeo, frame a frame: fica em `data/`, fora do Git, e não
pode ser publicado (RULES.md); a equipe o recebe por fora do repositório ou o regenera com os scripts.

**Passo 15 — Continuidade no tempo: Kalman, Viterbi e limiar mais baixo (`track_temporal.py`, `compare_gaps.py`).**
A fusão decide cada frame sozinha. Testamos usar os frames vizinhos, sobre os mesmos candidatos do jogo
todo (passo 13), em metros no campo:

- **Kalman:** rastro único com posição e velocidade; aceita só candidatos dentro da janela da previsão,
  segue a previsão por até 12 frames sem medida e recomeça numa bola confiante e coerente fora da janela.
- **Viterbi:** escolhe a sequência de menor custo no trecho inteiro (passado e futuro): custo 1 − conf por
  candidato, 0,3 pela bola do ATD, 0,5 por frame "sem bola", 0,1 por metro entre frames, saltos acima de
  45 m/s proibidos; lacunas de até 12 frames entre duas bolas são interpoladas. Uma detecção isolada só
  entra com conf > 0,8 (como na fusão), mas uma sequência coerente entra com conf média (~0,56 em 5 frames).

| | Fusão campeã | Kalman | Viterbi |
|---|---|---|---|
| Acerto, teste (79 frames com bola) | 67,1% | 74,7% | 79,7% |
| Acerto onde o ATD tem bola, teste | 92,0% | 84,0% | 92,0% |
| Acerto onde o ATD não tem bola, teste (29) | 24,1% | 58,6% | 58,6% |
| Falso alarme, teste (14 frames sem bola) | 7,1% | 42,9% | 42,9% |
| Falso alarme, 292 frames (33) | 21,2% | 57,6% | 48,5% |
| F1, 93 / 292 frames | 76,8% / 74,5% | 75,6% / 72,3% | 81,3% / 75,4% |

O Kalman piorou onde o ATD tem bola (o rastro "gruda" em objetos errados) e teve o menor F1. O Viterbi
acertou mais, mas a análise frame a frame dos 29 frames de teste sem ATD mostrou de onde vem o ganho e o
custo: dos 17 acertos dele, 16 são detecções do próprio Roboflow com confiança abaixo de 0,8 (que a fusão
descarta) e só 1 é interpolação; e o falso alarme também vem de detecções (4 dos 5 frames sem bola em que
ele deu posição), com 3 posições a 760–958 px da bola. Para separar o efeito da continuidade do efeito de
simplesmente aceitar confiança menor, rodamos a fusão com `--solo-conf 0,5` nos 292 frames: ela levou a maior
parte do ganho do Viterbi onde o ATD não tem bola (42,9% contra 49,0%; o campeão faz 31,6%), com MENOS falso
alarme (36,4% contra 48,5%). Ou seja, a continuidade quase não filtrou erros: Viterbi, limiar 0,5 e a fusão
campeã são pontos da mesma troca entre acerto e falso alarme, com F1 empatado dentro das margens (73,6% a
75,4% nos 292 frames). Ficamos com a fusão campeã, o ponto conservador: para a posse, uma bola inventada
(em jogo parado, ou ao lado do jogador errado) tende a custar mais que uma bola faltando, que a camada de
posse pode atravessar pela estabilidade da posse no tempo.

Essa foi a conclusão da etapa de detecção. Depois, na camada de posse, o rastro do Viterbi se mostrou a melhor
entrada: bola presente em 83,5% dos frames avaliados (72,8% na fusão), menos lacunas longas com o jogo rolando
e melhor validação dos modelos de posse (`posse_ml/README.md`, seções 5.2, 5.3 e 5.5). A submissão usa o Viterbi;
o Kalman ficou descartado.

## 4. Resultados nos 292 frames anotados (medição)

259 frames com bola visível e 33 sem; tolerância de 40 px; margens de até ±6 pontos (A), ±10 (B),
±8 (C) e ±17 (D). "@" é a confiança mínima do candidato; as fusões não têm limiar.

| | ATD | WASB @0,2 | WASB @0,5 | Roboflow @0,8 | Roboflow + máscara 3 m @0,8 | Fusão ATD+WASB | Fusão ATD+RF 0,8 | Fusão ATD+RF, override | **Fusão ATD+RF, override, máscara 3 m** |
|---|---|---|---|---|---|---|---|---|---|
| Acerto (A) | 48,6% | 45,9% | 39,4% | 50,6% | 53,7% | 54,4% | 60,6% | 65,3% | **66,4%** |
| Achou longe (A) | 13,5% | 42,5% | 23,6% | 17,0% | 6,2% | 22,0% | 22,0% | 17,4% | 11,6% |
| Perdido (A) | 37,8% | 11,6% | 37,1% | 32,4% | 40,2% | 23,6% | 17,4% | 17,4% | 22,0% |
| Acerto até 10 px (A) | 22,0% | 45,9% | 39,4% | 50,6% | 53,7% | 46,3% | 55,6% | 61,0% | 61,8% |
| Erro mediano nos acertos (px) | 10,7 | 2,3 | 2,3 | 0,6 | 0,6 | 2,6 | 0,7 | 0,7 | 0,7 |
| Acerto onde o ATD não tem bola (B, n = 98) | 0% | 22,4% | 15,3% | 31,6% | 31,6% | 15,3% | 31,6% | 31,6% | 31,6% |
| Acerto onde o ATD tem bola (C, n = 161) | 78,3% | 60,2% | 54,0% | 62,1% | 67,1% | 78,3% | 78,3% | 85,7% | 87,6% |
| Falso alarme (D, n = 33) | 15,2% | 78,8% | 12,1% | 36,4% | 9,1% | 24,2% | 48,5% | 48,5% | 24,2% |
| F1 | 59,3% | 46,3% | 47,9% | 58,7% | 66,7% | 60,6% | 64,2% | 69,1% | **73,3%** |

A máscara de 2 m foi a adotada no modelo final; nessa medição, ela ficou acima da de 3 m.

## 5. Limitações

- **Amostras pequenas.** 292 + 93 frames de medição; as margens são largas, sobretudo no
  falso alarme (33 frames sem bola nos 292 e só 14 nos 93, margem de ±26 pontos): o teste mostra
  que o modelo não aumenta o falso alarme do ATD, não que os dois sejam iguais.
- **Detector de um frame.** O Roboflow não usa os frames vizinhos e vê a bola com confiança média em muitos
  frames que a fusão descarta. Aproveitá-los pela continuidade (Kalman, Viterbi) ou por um limiar menor
  aumentou o acerto, mas na mesma medida o falso alarme (passo 15).
- **Falso alarme residual.** Parte dele vem do próprio ATD (15,2%), que a fusão mantém onde o RF concorda
  com ele ou não tem confiança para contrariá-lo.
- **Bola no ar, oclusões e bola fora do campo.** A homografia vale para o chão; bola alta pode ser cortada
  pela máscara, e bola escondida não é detectável pelo vídeo.
- **Domínio.** O Roboflow foi treinado em vídeo de transmissão da Bundesliga, não na câmera tática; não
  houve nenhum ajuste ao nosso vídeo.

## 6. Próximos passos

1. Deixar a camada de posse lidar com as lacunas da bola (a posse é estável no tempo), em vez de prever a
   bola no tracker.
2. Tratar a bola alta: a homografia vale para o chão, então uma bola no ar é projetada mais longe do que está.

## 7. Reprodução

Ambiente do autor: Windows, PowerShell, Python 3.13. Dependências: `numpy`, `pandas`, `opencv-python`,
`torch`, `torchvision`, `ultralytics` e `hydra-core` (este só para o WASB). Para usar a GPU NVIDIA, o PyTorch
precisa ser a versão com CUDA (por exemplo `python -m pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu128`). Pesos em `tracking_bola/pesos/`:

```
pip install gdown
gdown -O tracking_bola/pesos/football-ball-detection.pt "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
```

Para repetir as comparações, clonar em `tracking_bola/third_party/` o WASB-SBDT
(`git clone https://github.com/nttcom/WASB-SBDT`, pesos do Google Drive indicado no README deles) e o
FootAndBall (`git clone https://github.com/jac99/FootAndBall`, pesos no próprio repositório). Todas as
saídas ficam em `tracking_bola/out/`, fora do Git.

Modelo final, nos 292 frames anotados (medição):

```
python tracking_bola/detect_yolo.py --weights tracking_bola/pesos/football-ball-detection.pt --class-id 0 --out tracking_bola/out/yolo_roboflow.csv
python tracking_bola/mask_pitch.py --detector tracking_bola/out/yolo_roboflow.csv --margin 2 --out tracking_bola/out/yolo_roboflow_campo2.csv
python tracking_bola/fuse_atd_wasb.py --wasb tracking_bola/out/yolo_roboflow_campo2.csv --solo-conf 0.8 --override-conf 0.8 --out tracking_bola/out/fusao_rf_campo2.csv
python tracking_bola/compare_trackers.py --tracker fusao_rf_campo2=tracking_bola/out/fusao_rf_campo2.csv --tol 40
python tracking_bola/show_false_alarms.py --tracker tracking_bola/out/fusao_rf_campo2.csv --detector tracking_bola/out/yolo_roboflow_campo2.csv
```

Jogo todo (modelo final; a primeira linha só mede o tempo; `--device 0` usa a GPU):

```
python tracking_bola/run_champion_full.py --limit-frames 200 --device 0
python tracking_bola/run_champion_full.py --device 0
python tracking_bola/export_ball.py
```

Conjunto de teste: anotar com `annotate_ball.py --start 300 --n 400 --out tracking_bola/out/annotations/ball_annotations_teste.csv`
e repetir os comandos acima passando `--annotations tracking_bola/out/annotations/ball_annotations_teste.csv`
ao `detect_yolo.py`, ao `fuse_atd_wasb.py` e ao `compare_trackers.py`.

Diagnósticos e modelos descartados:

```
python tracking_bola/gap_map.py
python tracking_bola/check_projection.py
python tracking_bola/eval_atd.py
python tracking_bola/check_atd_nature.py
python tracking_bola/measure_ball.py
python tracking_bola/detect_yolo.py --out tracking_bola/out/yolo_sports_ball.csv
python tracking_bola/diagnose_wasb.py --wasb-dir tracking_bola/third_party/WASB-SBDT --weights tracking_bola/pesos/wasb_soccer_best.pth.tar --scales 0.4 0.5 0.6 0.7 0.75 0.8 0.9 --frames 100 --images 0
python tracking_bola/detect_wasb.py --wasb-dir tracking_bola/third_party/WASB-SBDT --weights tracking_bola/pesos/wasb_soccer_best.pth.tar --scale 0.6 --out tracking_bola/out/wasb_060.csv
python tracking_bola/fuse_atd_wasb.py --wasb tracking_bola/out/wasb_060.csv --out tracking_bola/out/fusao_wasb.csv
python tracking_bola/track_temporal.py --method kalman
python tracking_bola/track_temporal.py --method viterbi
python tracking_bola/compare_gaps.py
python tracking_bola/detect_footandball.py --fb-dir tracking_bola/third_party/FootAndBall
python tracking_bola/diagnose_footandball.py --fb-dir tracking_bola/third_party/FootAndBall
```

As anotações dependem do operador: a semente e a lista de frames sorteados são determinísticas, mas os
cliques não são reproduzíveis por outra pessoa. Elas servem só para validar e testar; o sistema não
depende delas para produzir sua saída.

## Referências

- Roboflow, *sports* (github.com/roboflow/sports) e *Ball Tracking in Sports with Computer Vision* (blog Roboflow).
- Tarashima et al., *Widely Applicable Strong Baseline for Sports Ball Detection and Tracking* (WASB-SBDT), BMVC 2023, arXiv:2311.05237.
- Komorowski, Kurzejamski, Sarwas, *FootAndBall: Integrated Player and Ball Detector*, VISAPP 2020, arXiv:1912.05445.
- *TOTNet: Occlusion-Aware Temporal Tracking for Robust Ball Detection in Sports Videos*, arXiv:2508.09650.
- D'Orazio et al., ISSIA-CNR Soccer Dataset.