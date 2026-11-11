# posse_ml/

Rotulagem de posse frame a frame (H, A ou D) com classificadores treinados nos jogos abertos da
DFL, usando a **nossa bola** no lugar da bola do ATD. Diferente de `posse/`, que só decide H ou A
dentro da bola viva, aqui a bola morta também pode ser decidida junto com a posse, ou vir do
detector de paradas da equipe (`analise/out/paradas_D.csv`).

Resumo do estado atual (seção 6): bola do Viterbi do tracking, perfil da bola medido no jogo
inteiro, bola morta do detector de paradas em modo **suave** ou **rígido**. Na janela pública:
spellF1 0,690 nos dois; macroF1 0,941 (suave) e 0,931 (rígido).

## 1. Como funciona

1. **Partida** (`partida.py`). Jogadores do ATD, bola do nosso CSV (pixels viram campo pela
   homografia), tudo em metros. O time dos
   dois slots de goleiro é corrigido frame a frame pelo lado do campo, com a mesma regra do
   `posse/predictor.py` (seção 1.3 do `posse/README.md`).
2. **Bola** (`bola.py`). Filtro de pulos (3 m da mediana de ±5 frames, o mesmo do `posse/`), ponte
   linear nas lacunas de até 8 frames e velocidade por diferença central de ±2 frames.
3. **Atributos** (`atributos.py`, 100 colunas). Nada usa identidade de jogador: o mais perto de cada
   time, quantos a 2 m e a 5 m, para onde a bola vai em 0,5 s e 1 s, há quanto tempo a bola não é
   vista, ritmo dos jogadores, saldo de proximidade antes e depois, domínio claro, e a forma de cada
   time no próprio sentido de ataque (centro, espalhamento, última linha, avanço, marcação), mais a
   tendência dessas grandezas em 2 s. A forma é o que sobra quando o tracker perde a bola, que no
   nosso jogo ainda acontece em mais de um quarto dos frames avaliados.
4. **Treino na DFL** (`dfl.py`, `degradacao.py`, `treinar.py`). Os sete jogos abertos da DFL têm, por
   frame, o status da bola e o time com a posse. Cada jogo é **degradado até parecer o nosso**:
   ruído e saídas nos jogadores, bola no ar projetada no chão pela câmera tática, a bola da parada
   refeita (a DFL a congela e teleporta), e lacunas e pulos da bola **com as estatísticas medidas no
   nosso tracker** (`perfil.py`, `medir_bola.py`): fração de bola faltando com o jogo rolando e
   parado, histograma do tamanho das lacunas em cada caso, taxa e tamanho dos pulos, posição da
   câmera. O perfil é medido **no jogo inteiro**, com viva/morta pelo detector de paradas da equipe
   (seção 5.1 explica por que não nos 5 minutos do GT). Os modelos estão descritos logo abaixo, em
   "Os classificadores".
5. **Transmissão** (`transmissao.py`). O vídeo de TV é alinhado ao tático (um deslocamento por
   tempo, achado por correlação cruzada das bolas) e dá o plano de cada frame: ao vivo, outra vista,
   ou sem campo (replay, close). Sem campo é forte evidência de bola morta.
6. **Viterbi** (`viterbi.py`). Quatro estados: viva com o mandante, viva com o visitante, morta com o
   mandante para reiniciar, morta com o visitante. Cada troca tem preço, e de "morta com X" só se
   sai para "viva com X". Antes do kickoff a bola é morta; no segundo depois dele, viva.
7. **Calibração na janela pública** (`calibrar.py`). Só o que a DFL não ensina: o peso da
   transmissão (e, no modo suave, do detector de paradas) na bola morta, e quanto adiantar as
   bordas das paradas, que com a nossa bola tendem a sair atrasadas.
8. **Bola morta de fora** (opcional, `--paradas`). No modo **rígido**, a bola morta é a do detector
   de paradas, e os modelos só decidem de quem é a bola viva e quem repõe a parada. No modo
   **suave**, o detector é mais uma evidência de bola morta, somada ao modelo e à TV.

### Os classificadores

Por padrão, três `HistGradientBoostingClassifier` do scikit-learn (`modelos.py`; os outros estão na
subseção seguinte), cada um treinado só nas linhas em que a pergunta dele faz sentido:

| modelo | pergunta | linhas de treino |
|---|---|---|
| `morta` | a bola está parada neste frame? | todos os frames com rótulo |
| `time` | com a bola rolando, é do mandante (Barça)? | só bola viva |
| `reinicio` | com a bola parada, o mandante vai repor? | só bola parada com reinício conhecido |

Hiperparâmetros, iguais nos três: 400 iterações (árvores), taxa de aprendizado 0,05, até 63 folhas
por árvore, no mínimo 200 amostras por folha, regularização L2 de 1,0, sem parada antecipada. Os
valores ausentes (bola sumida, jogador fora do tracking) vão direto para o modelo, que trata NaN
sozinho.

- **Dados:** os 7 jogos da DFL, 3 cópias degradadas de cada um, um frame a cada 2: cerca de 1 milhão
  de linhas e 100 colunas por conjunto.
- **Dois conjuntos:** "direto" (duas cópias na orientação original) e "invertido" (uma original e
  uma com os lados trocados, para o modelo não aprender que o mandante tem mais a bola). A previsão
  é a média dos dois em log-odds.
- **Validação** (escolha dos custos do Viterbi): deixando um jogo de fora por vez. Com `--rapido`,
  150 árvores e um frame a cada 4; os modelos finais não mudam.

Os modelos ficam em `modelos/<instância>/conjuntos/{direto,invertido}/` como pickles, com um
manifesto da ordem das colunas. Por isso o `requirements.txt` fixa a versão do scikit-learn.

### Outros classificadores (`--classificador`)

O HistGradientBoosting é o padrão. Os outros usam as mesmas 100 colunas, as mesmas cópias
degradadas, os mesmos dois conjuntos e o mesmo decodificador; muda só o modelo que dá as três
probabilidades.

| `--classificador` | o que é | configuração |
|---|---|---|
| `hgb` (padrão) | HistGradientBoosting (scikit-learn) | a de cima |
| `xgboost` | XGBoost, árvores crescendo por folha (`lossguide`) | 400 árvores, taxa 0,05, até 63 folhas, `min_child_weight` 10, L2 1 |
| `lightgbm` | LightGBM | 400 árvores, taxa 0,05, 63 folhas, no mínimo 200 amostras por folha, L2 1 |
| `catboost` | CatBoost, árvores simétricas | 400 árvores, taxa 0,05, profundidade 6, L2 1 |
| `gru` | rede recorrente: GRU bidirecional de 2 camadas | largura 64, 10 épocas |
| `lstm` | rede recorrente: LSTM bidirecional de 2 camadas (a GRU com uma memória a mais) | largura 64, 10 épocas |
| `tcn` | rede convolucional temporal não causal: 6 blocos residuais, dilatação 1 a 32 | largura 64, 10 épocas; enxerga ~10 s para cada lado |
| `transformer` | encoder de 3 camadas, 4 cabeças, viés de posição relativa, convolução na entrada | largura 64, 10 épocas, taxa 0,001; atenção sobre a janela de ~20 s |

As quatro de árvore treinam os três modelos (`morta`, `time`, `reinicio`) como acima. Nelas o
contexto só entra pelas colunas de janela (médias de 2 s antes e depois, "desde" e "até").

As redes (`sequencia.py`) leem o trecho inteiro e aprendem sozinhas o que importa antes e
depois de cada frame. Uma rede só responde às três perguntas, com três saídas; a perda de cada uma
só conta nos frames em que ela faz sentido (as mesmas linhas da tabela de cima). Detalhes:

- as colunas são padronizadas com a média e o desvio do treino; NaN vira 0, com uma coluna 0/1 de
  "faltando" para cada atributo que às vezes falta;
- um frame a cada 2 (12,5 Hz). Na previsão, a rede roda nos frames pares e nos ímpares, e todo
  frame recebe a sua probabilidade;
- treino em janelas de 256 passos (~20 s) sorteadas das cópias, com AdamW e taxa em ciclo único;
  previsão em janelas com metade de sobreposição, ficando com o miolo de cada uma;
- `--gpu` treina na placa (PyTorch com CUDA). A previsão roda sempre na CPU, para a submissão sair
  igual em qualquer máquina;
- com `--rapido`, a validação usa 5 épocas em vez de 10.

**O transformer** segue o que funciona em segmentação temporal de ações, o problema mais parecido
com o nosso (um rótulo por frame a partir de atributos por frame, com segmentos longos e trocas
raras):

- **atenção local:** cada frame só olha a janela de treino (~20 s), não o jogo inteiro. Atenção
  sobre a sequência toda é cara e, com poucos dados, rende menos que a local;
- **posição relativa:** em vez de marcar a posição absoluta de cada frame, cada cabeça aprende um
  peso para cada distância entre dois frames ("o que aconteceu 2 s antes"). Nos testes de
  segmentação, a relativa ganhou da absoluta;
- **convolução na entrada:** transformer puro não sabe que frames vizinhos se parecem e precisa de
  muito dado para aprender isso; uma convolução de 5 passos antes da atenção dá esse prior local;
- **pequeno e com normalização antes de cada subcamada (pre-LN):** ~136 mil parâmetros, como a GRU
  e a TCN, e taxa de aprendizado menor (0,001), que treinam com mais estabilidade com poucos dados;
- **sem CRF nem perda de suavização:** o Viterbi já faz esse papel (troca só com evidência
  sustentada), e num trabalho recente de eventos de posse na DFL a decodificação com restrições foi
  responsável pela maior parte do ganho do CRF.

O que esperar: em segmentação de ações, transformers bem desenhados passam a TCN nos conjuntos
maiores e perdem nos pequenos. No trabalho sobre posse na DFL, atenção e LSTM bidirecional ficaram
praticamente empatadas, porque a posse depende mais do contexto de curto prazo. Referências:
ASFormer (Yi et al., BMVC 2021, arXiv:2110.08568); "Do we really need temporal convolutions in
action segmentation?" (Du et al., arXiv:2205.13425); PathCRF (arXiv:2602.12080).

Cada classificador grava em `modelos/<instância>/<classificador>/` e a submissão sai como
`posse_ml_<instância>_<classificador>.csv` (o `hgb` continua em `modelos/<instância>/` e
`posse_ml_<instância>.csv`). O decodificador e a calibração são próprios de cada um.

### Regras de posse (o que o decodificador acaba fazendo)

Não há regra escrita à mão: elas saem do preço por troca do Viterbi.

- **Trocar exige evidência sustentada.** A posse só passa para o outro time quando os frames
  seguintes, somados, favorecem esse time o bastante para pagar a troca. Enquanto a bola viaja ou
  some, os frames quase não pesam para nenhum lado e a posse fica onde estava.
- **Dividida não troca.** Com gente dos dois times junto da bola, cada frame fica perto do empate.
- **Toque não é posse.** Um desvio ou um toque que não segura a bola dá poucos frames a favor do
  adversário, que não pagam a troca. Se o adversário sai com a bola, os frames seguintes pagam.
- **Na parada ninguém tem a bola**, mas o estado de parada já diz quem vai repor (pelo jogador junto
  da bola), e a posse depois do reinício é desse time.

## 2. Como rodar

Da raiz do repositório (PowerShell). Python 3.12 ou mais novo.

### Instâncias: uma por bola

Cada bola tem a sua instância, com perfil, modelos e submissão próprios. O treino imita a bola
medida, então os modelos de uma bola não servem para outra.

| instância | bola | modelos | submissão |
|---|---|---|---|
| `viterbi` (padrão) | `tracking_bola/out/viterbi_bola.csv` | `modelos/viterbi/` | `evaluation/results/posse_ml_viterbi.csv` |
| `fusao` | `data/tactical/BAR-ATH_tactical_ball_tracker.csv` | `modelos/fusao/` | `evaluation/results/posse_ml_fusao.csv` |
| `atd` | a bola do ATD | `modelos/atd/` | `evaluation/results/posse_ml_atd.csv` |

```powershell
python -m pip install -r posse_ml/requirements.txt
python -m posse_ml.baixar_dfl                       # uma vez, ~2,5 GB em posse_ml/dfl/

# para cada instância (troque viterbi por fusao ou atd)
python -m posse_ml.medir_bola --instancia viterbi   # perfil da bola, viva/morta pelo paradas_D.csv
python -m posse_ml.treinar --rapido --instancia viterbi
python -m posse_ml.calibrar --instancia viterbi
python -m posse_ml --instancia viterbi

# a candidata atual (seção 6): bola morta do detector de paradas
python -m posse_ml.calibrar --instancia viterbi --paradas analise\out\paradas_D.csv
python -m posse_ml --instancia viterbi --paradas analise\out\paradas_D.csv --modo-paradas suave
python -m posse_ml --instancia viterbi --paradas analise\out\paradas_D.csv          # rígido

# comparar
python -m posse_ml.comparar_bolas                   # as bolas entre si, sem modelo nenhum
python -m posse_ml.comparar_paradas                 # as fontes de bola morta contra o GT público
python evaluation/run_evaluation.py --bootstrap 1000
```

O `medir_bola` separa bola viva de morta, no jogo inteiro, com o detector de paradas da equipe
(`analise/out/paradas_D.csv`); sem ele, usa os 5 minutos do GT público, que têm poucas lacunas.

### Bola morta de fora (`--paradas`)

A bola morta pode vir de outra fonte, como o detector de paradas da equipe
(`analise/out/paradas_D.csv`, frame,morta) ou, só para teste, o GT público (frame,label). Os modelos
são os mesmos; muda o decodificador.

```powershell
python -m posse_ml.comparar_paradas                          # as fontes de bola morta contra o GT público

# teto: bola morta perfeita na janela pública (usa o GT; nunca enviar)
python -m posse_ml --paradas evaluation\gt\BAR-ATH_possession_gt_public.csv

# rígido: a bola morta é a do detector; o modelo só decide de quem é a bola e quem repõe
python -m posse_ml --paradas analise\out\paradas_D.csv

# suave: o detector é mais uma evidência de bola morta, com peso ajustado na janela pública
python -m posse_ml.calibrar --paradas analise\out\paradas_D.csv
python -m posse_ml --paradas analise\out\paradas_D.csv --modo-paradas suave
```

Cada um grava `posse_ml_<instância>_<fonte>_<modo>.csv` (por exemplo `posse_ml_viterbi_nuno_rigido.csv`),
e o modo suave usa `modelos/<instância>/calibracao_<fonte>.json`, sem mexer no `calibracao.json`.

### Outros classificadores

Tudo de uma vez: treina cada classificador que ainda não está treinado (na GPU quando dá: xgboost,
catboost, gru, lstm e tcn; o hgb e o lightgbm ficam na CPU), calibra, grava as três submissões de cada um
(bola morta própria, detector rígido, detector suave) e monta a tabela de comparação.

```powershell
python -m pip install -r posse_ml/requirements-classificadores.txt   # uma vez
# redes na GPU: instale o PyTorch com CUDA pelo seletor de pytorch.org (o do PyPI é só CPU no Windows)

python -m posse_ml.comparar_classificadores                       # todos: hgb, xgboost, lightgbm, gru, tcn, catboost, lstm, transformer
python -m posse_ml.comparar_classificadores --classificadores xgboost tcn
python -m posse_ml.comparar_classificadores --so-comparar         # não treina; só refaz a tabela
python -m posse_ml.comparar_classificadores --retreinar xgboost   # treina de novo (sem nome: todos)
```

A tabela sai na tela e em `modelos/<instância>/comparacao_classificadores.md`; o log de cada treino
fica em `cache/logs/`. Se um classificador falhar (pacote faltando, memória), os outros seguem. A
validação é a rápida; `--completo` usa a inteira. O jogo é lido e os atributos calculados uma vez só.

**Combinar classificadores** (média dos log-odds), sem treinar nada:

```powershell
python -m posse_ml.combinar                                   # gru + tcn
python -m posse_ml.combinar --classificadores gru tcn xgboost
python -m posse_ml.combinar --so-validacao                    # só a tabela da DFL
```

A validação usa as probabilidades que o treino de cada membro deixou em `cache/validacao/`: em cada
jogo de fora, a combinação é a média das previsões dos membros para aquele jogo, que nenhum deles viu
no treino. Os custos do Viterbi são escolhidos de novo para a combinação, com a mesma grade e a
mesma regra. A tabela mostra o spellF1 de cada jogo para os membros e para a combinação. Depois, a
combinação é gravada em `modelos/<instância>/<a+b>/` (`membros.json`, `decodificador.json`,
calibrações), calibrada, e as três submissões saem como `posse_ml_<instância>_<a+b>_...csv`.

**Testar sementes** (o resultado de uma rede depende do sorteio inicial):

```powershell
python -m posse_ml.testar_sementes                              # tcn e gru, sementes 1 e 2
python -m posse_ml.testar_sementes --classificadores tcn --sementes 1 2 3 4
python -m posse_ml.testar_sementes --so-comparar                # não treina; só refaz a tabela
```

A semente 0 é o modelo que já existe. Cada semente nova muda só o sorteio do modelo (pesos
iniciais, ordem das janelas, dropout), com `--hiper semente=N`; as cópias da DFL são as mesmas, do
cache. Os modelos ficam em `modelos/<instância>/<clf>_s<N>/`, e a média das sementes de cada
classificador (`<clf>_sementes`) é validada na DFL e gravada como uma combinação. Para cada semente
e para a média sai a validação na DFL, as três versões na janela e o teto (bola morta do GT, sem
gravar arquivo). A tabela fica em `modelos/<instância>/sementes.md`. Na GPU, ~5 min por semente.

**Testar variantes das redes** (número de épocas, a LSTM e o transformer):

```powershell
python -m posse_ml.testar_variantes                       # todas; treina só as que faltam
python -m posse_ml.testar_variantes --variantes tcn_e5 lstm_e5 lstm_e20
python -m posse_ml.testar_variantes --so-comparar
```

| variante | o que muda | épocas (validação / final) |
|---|---|---|
| `lstm` | LSTM no lugar da GRU, mesma configuração | 5 / 10 |
| `lstm_e5` | LSTM com 5 épocas | 5 / 5 |
| `lstm_e20` | LSTM com 20 épocas | 20 / 20 |
| `tcn_e5` | TCN com 5 épocas | 5 / 5 |
| `tcn_e10` | nada no modelo final; a validação usa as mesmas 10 épocas (a `tcn` padrão valida com 5) | 10 / 10 |
| `tcn_e20` | 20 épocas | 20 / 20 |
| `tcn_e30` | 30 épocas | 30 / 30 |
| `transformer` | o transformer, configuração padrão | 5 / 10 |

A `tcn` e a `gru` que já existem entram como referência, sem treinar de novo. Saem a validação na
DFL, as três versões na janela, o teto e duas medidas por frame na janela, tiradas do modelo antes
do Viterbi: acerto de bola morta/viva e acerto de H/A com a bola viva. São ~7.500 frames em vez de 17
posses, então oscilam bem menos que o spellF1 (mas frames vizinhos não são independentes). A tabela
fica em `modelos/<instância>/variantes.md`. Na GPU, ~15 min por variante de 5 a 10 épocas e ~30 a
40 min para 20 ou 30.

**Testar épocas com várias sementes** (validação sempre com 5 épocas; muda só o modelo final):

```powershell
python -m posse_ml.testar_epocas                      # lstm, transformer e tcn; 5, 10 e 20 épocas; sementes 0, 1 e 2
python -m posse_ml.testar_epocas --classificadores tcn --epocas 5 10 --sementes 0 1
python -m posse_ml.testar_epocas --sementes 10 11 12       # sementes novas, sem reaproveitar nada
python -m posse_ml.testar_epocas --classificadores tcn lstm --epocas 1 2 3 5 --sementes 10 11 12 --validacao igual
python -m posse_ml.testar_epocas --so-comparar
```

Os modelos ficam em `modelos/<instância>/<rede>_f<N>[_s<semente>]`; os de 10 épocas usam os nomes do
`testar_sementes` (`tcn`, `tcn_s1`...) e são reaproveitados quando já existem. A validação de uma
semente é a mesma para todas as épocas e sai uma vez só (do cache, quando já existe), então a DFL não
separa as épocas: quem compara é a janela, com a média das sementes, e as medidas por frame. A
tabela traz, para cada rede e número de épocas, o valor de cada semente e a média deles, e fica em
`modelos/<instância>/epocas.md`. Com `--validacao igual`, a validação usa as mesmas épocas do modelo
final (como na TCN-5 campeã), os modelos ficam em `<rede>_f<N>v<N>[_s<semente>]` (o de 5 épocas é o
mesmo `<rede>_f5`, reaproveitado) e a tabela vai para `epocas_igual.md`. As 27 combinações levam ~3 horas na GPU, a maior parte nas de 20
épocas; o que já existe é pulado, e o comando pode ser repetido se for interrompido.

**Testar os dados de treino: só DFL, só Metrica, DFL + Metrica**

```powershell
python -m posse_ml.baixar_metrica      # uma vez, ~190 MB em posse_ml/metrica_dados
python -m posse_ml.metrica             # confere os rótulos derivados ao lado do GT público
python -m posse_ml.testar_dados        # tcn e lstm, 5 épocas, sementes 10 11 12
python -m posse_ml.testar_dados --dados metrica --epocas 12       # só um treino, outro nº de épocas
python -m posse_ml.testar_dados --dados dfl+metrica --epocas 3
```

Os 3 jogos abertos da Metrica (github.com/metrica-sports/sample-data) trazem rastreamento a 25 Hz
e eventos anotados, mas não a posse frame a frame. Ela é derivada em `metrica.py`:

- **bola morta:** a bola desses dados é reconstruída dos eventos (anda em linha reta nos passes,
  nunca falta durante um evento) e some quando o jogo para; bola ausente por 1 s ou mais é `D`;
- **posse:** o time do último evento com a bola (passe, condução, finalização, bola parada,
  recuperação, perda); a perda continua com quem perdeu até o adversário recuperar;
- **toque sem controle:** um trecho de menos de 2 s sem passe, finalização nem bola parada não
  conta como posse. O limite de 2 s foi escolhido para a duração das posses ficar como no GT.

| | bola morta | posses/min | duração média | mediana / 10% mais curtas |
|---|---|---|---|---|
| GT público (5 min) | 19,6% | 3,40 | 14,2 s | 7,6 s / 2,2 s |
| MET1 | 37,7% | 3,13 | 12,0 s | 7,5 s / 2,2 s |
| MET2 | 40,4% | 3,00 | 11,9 s | 7,1 s / 1,6 s |
| MET3 | 31,6% | 3,30 | 12,4 s | 7,4 s / 1,9 s |

Para comparar: a DFL tem ~5 posses por minuto, uma convenção mais picada que a do desafio. No treino
(`treinar --dados metrica` ou `--dados dfl+metrica`), os jogos da Metrica passam pela mesma
degradação da DFL: uma bola de parada sintética na bola morta e as falhas do perfil da nossa bola.
O `testar_dados` treina, para cada rede e semente, os três casos com a mesma configuração e compara
na janela pública (`modelos/<instância>/dados.md`); o caso "dfl" é o `<rede>_f5_s<semente>` que já
existe. Limitações: os rótulos são derivados por regra, a bola é mais "limpa" que uma rastreada, e o
limite de 2 s foi ajustado olhando a distribuição do GT público (só a duração das posses, não o
placar).

**Testar outras configurações da TCN e da LSTM** (com sementes; o número de épocas é livre):

```powershell
python -m posse_ml.testar_configs                                    # todas, 5 épocas, sementes 10 11 12
python -m posse_ml.testar_configs --epocas 3                         # outro número de épocas
python -m posse_ml.testar_configs --configs tcn_d03 lstm_c1 --epocas 3 5
python -m posse_ml.testar_configs --so-comparar
```

| configuração | o que muda (o resto fica no padrão) |
|---|---|
| `tcn_d03`, `lstm_d03` | dropout 0,3 (padrão 0,1): regulariza, pode proteger da especialização na DFL |
| `tcn_l32`, `lstm_l32` | largura 32 (padrão 64) |
| `tcn_l128`, `lstm_l128` | largura 128 |
| `tcn_b4` | 4 blocos: alcance de ~±2,5 s (padrão 6 blocos, ~±10 s) |
| `tcn_b7` | 7 blocos e janela de 512 passos: alcance de ~±20 s |
| `lstm_c1`, `lstm_c3` | 1 ou 3 camadas (padrão 2) |

A validação usa as mesmas épocas do modelo final (`--validacao igual`, padrão; `--validacao 5` para
sempre 5). Os modelos ficam em `modelos/<instância>/<config>_f<N>[v<N>]_s<semente>`, e a TCN e a
LSTM padrão com as mesmas épocas e sementes (do `testar_epocas`) entram como referência. A tabela
(média das sementes e o valor de cada uma) fica em `modelos/<instância>/configs.md`. As 10
configurações com 3 sementes são 30 modelos, ~3 horas na GPU com 5 épocas.

Essas opções também valem no `--hiper` do `treinar`: `largura`, `dropout`, `blocos` (TCN), `camadas`
(GRU, LSTM, transformer) e `janela` (passos de cada trecho de treino e previsão, padrão 256 = ~20 s; o
alcance da TCN não deve passar da janela).

**Testar árvores com menos árvores e/ou mais rasas** (o equivalente a treinar menos):

```powershell
python -m posse_ml.testar_arvores                          # HGB: 50, 150 e 400 árvores x 15 e 63 folhas
python -m posse_ml.testar_arvores --arvores 25 50 --folhas 7 15
python -m posse_ml.testar_arvores --classificador xgboost  # também xgboost e lightgbm
python -m posse_ml.testar_arvores --so-comparar
```

Cada configuração fica em `modelos/<instância>/<classificador>_a<árvores>_f<folhas>/`. A validação usa
o mesmo número de árvores do modelo final (com `--hiper max_iter=N`, o `treinar --rapido` deixa de
trocar por 150); continua rápida só em usar um frame a cada 4. Árvore não depende de semente: uma
rodada por configuração. A configuração padrão entra como referência, e a tabela fica em
`modelos/<instância>/arvores.md`.

Um classificador só, passo a passo:

```powershell
python -m posse_ml.treinar --rapido --instancia viterbi --classificador xgboost --gpu
python -m posse_ml.calibrar --instancia viterbi --classificador xgboost
python -m posse_ml --instancia viterbi --classificador xgboost
python -m posse_ml.calibrar --instancia viterbi --classificador xgboost --paradas analise\out\paradas_D.csv
python -m posse_ml --instancia viterbi --classificador xgboost --paradas analise\out\paradas_D.csv --modo-paradas suave
python -m posse_ml --instancia viterbi --classificador xgboost --paradas analise\out\paradas_D.csv
```

As cópias degradadas da DFL vão para o cache no primeiro treino e servem a todos os classificadores
da mesma instância; os outros não refazem essa parte.

**Tempo das redes (estimativa).** Na DFL, uma época dá ~100 passos de 32 janelas. Medido em 2
núcleos de CPU: ~0,5 s por passo na GRU e ~0,16 s na TCN, ou seja, ~8 e ~3 minutos por rede de 10
épocas. A validação com `--rapido` treina 14 redes de 5 épocas: perto de 1 hora na GRU e 20 minutos
na TCN, em CPU, e bem menos com `--gpu`.

**Nome do arquivo para o formulário:** o SUBMISSION.md pede `<nome_da_equipe>.csv`. Na hora de
enviar, copie a submissão escolhida com esse nome e rode o `validate_submission.py` nela.

**Tempo (estimativa).** Num teste com dados sintéticos do mesmo tamanho, o HistGradientBoosting
levou cerca de 0,045 s por árvore a cada 100 mil linhas, em 2 núcleos. Isso dá uns 6 minutos por conjunto de três
modelos em 2 núcleos, menos num notebook com mais núcleos. Somando as sete rodadas de validação
(dois conjuntos cada) e os modelos finais, o treino completo fica na casa de 1 hora; com `--rapido`,
uns 20 a 30 minutos. A primeira leitura de cada jogo da DFL e a degradação das cópias também levam
alguns minutos, mas vão para o cache. `--passo 4` (um frame a cada 4) corta o treino pela metade. A
submissão leva poucos minutos com o ATD já no cache.

| comando | argumentos úteis |
|---|---|
| `python -m posse_ml` | `--instancia`, `--classificador`, `--paradas CSV`, `--modo-paradas rigido|suave`, `--sem-transmissao`, `--sem-goleiros`, `--out`, `--bola CSV` |
| `posse_ml.medir_bola` | `--instancia`, `--estados` (viva/morta: frame,label ou frame,morta), `--sem-camera` |
| `posse_ml.treinar` | `--instancia`, `--classificador`, `--gpu`, `--rapido`, `--so-validacao`, `--pular-validacao`, `--passo`, `--processos 2` (só com RAM sobrando), `--hiper max_iter=600` (redes: `--hiper epocas=15,largura=96`) |
| `posse_ml.calibrar` | `--instancia`, `--classificador`, `--paradas CSV`, `--sem-transmissao` |
| `posse_ml.comparar_classificadores` | `--instancia`, `--classificadores`, `--so-comparar`, `--retreinar`, `--completo`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.combinar` | `--instancia`, `--classificadores` (também `tcn_s1` etc.), `--so-validacao`, `--paradas CSV` |
| `posse_ml.testar_dados` | `--instancia`, `--classificadores`, `--epocas`, `--sementes`, `--so-comparar`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.testar_configs` | `--instancia`, `--configs`, `--epocas`, `--validacao igual\|5`, `--sementes`, `--so-comparar`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.testar_arvores` | `--instancia`, `--classificador`, `--arvores`, `--folhas`, `--so-comparar`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.testar_epocas` | `--instancia`, `--classificadores`, `--epocas`, `--sementes`, `--so-comparar`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.testar_variantes` | `--instancia`, `--variantes`, `--so-comparar`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.testar_sementes` | `--instancia`, `--classificadores`, `--sementes`, `--so-comparar`, `--completo`, `--sem-gpu`, `--paradas CSV` |
| `posse_ml.comparar_bolas` | nenhum |
| `posse_ml.comparar_paradas` | `--arquivos` (outras fontes de bola morta) |

## 3. Problemas comuns no Windows

- **`No module named 'numpy'`** ou outra versão do Python: o `python` da janela pode apontar para
  outro Python (pode haver mais de um instalado; as dependências estão no 3.13). Rode
  `python -c "import sys; print(sys.executable)"` e, se não for o 3.13,
  `Set-Alias python "$HOME\AppData\Local\Programs\Python\Python313\python.exe"` (vale para a
  janela aberta). Mais simples: trocar `python` por `py -3.13` em todos os comandos (o lançador do
  Python escolhe a versão), por exemplo `py -3.13 -m posse_ml.testar_variantes`.
- **A tabela do `run_evaluation` some da tela:** com dezenas de submissões em `evaluation\results`,
  o bloco de detalhes empurra a tabela para cima e o `--bootstrap` demora. Veja "Comparar no
  `run_evaluation`" na seção 4.
- **Memória** (`Unable to allocate ...`): o treino precisa de ~1,5 GB livres no pico. Feche o
  navegador e o viewer. `--processos` é 1 por padrão; 2 só com RAM sobrando. Se ainda faltar,
  aumente o arquivo de paginação do Windows (Memória virtual, em Configurações avançadas do
  sistema).
- **O treino parece parado** depois de "perfil da bola": está degradando as 21 cópias da DFL
  (7 jogos x 3), uma linha por cópia pronta. Com o perfil já usado antes, sai do cache em segundos.
- **Treino interrompido:** rode o mesmo comando. Cópias prontas e rodadas de validação terminadas
  ficam no cache.
- **Download da DFL travado:** o `baixar_dfl` mostra o progresso a cada 15 s, reconecta sozinho e
  continua os `.parcial`. O tamanho de arquivo aberto no Explorer ou no `Get-ChildItem` não
  atualiza enquanto ele baixa; não use isso para medir o progresso.
- **Zip baixado de novo:** o Windows salva como `posse_ml (N).zip` (`ls $HOME\Downloads\posse_ml*`
  mostra o mais recente). Descompacte por cima
  (`Expand-Archive ... -DestinationPath . -Force`) sem apagar `posse_ml\dfl` nem `posse_ml\modelos`.

## 4. Como ler os números

- **Janela pública = 5 minutos, 17 posses.** Uma posse a mais ou a menos mexe ~0,06 no spellF1. O
  bootstrap do `run_evaluation` dá intervalo de ~0,45 a ~0,85 para o spellF1 de todas as versões do
  `posse_ml`. A janela serve para ver se o pipeline está correto e para achar erros grandes, não
  para escolher entre versões parecidas.
- **O spellF1 do jogo inteiro não tem como ser medido** antes do resultado: o GT completo é privado.
- **Validação na DFL** (tabela do `treinar`): jogo inteiro, em jogos que o modelo nunca viu, mas com
  o rótulo da DFL, que troca de posse com muito mais frequência que o do desafio. Não é a mesma
  régua da janela.
- **Ajuste na janela:** o peso da TV, o do detector de paradas (modo suave) e o adiantamento das
  paradas são ajustados nos mesmos 5 minutos em que são avaliados. Isso é permitido (o GT público
  foi publicado para isso), mas deixa o número da janela um pouco otimista.

| versão | ajustado na janela pública |
|---|---|
| bola morta própria (`posse_ml_viterbi`) | pesos da TV, adiantamento |
| detector de paradas, suave | pesos da TV e do detector, adiantamento |
| detector de paradas, rígido | nada (os custos do Viterbi vêm da DFL) |

### Comparar no `run_evaluation`

Os scripts de teste (`comparar_classificadores`, `combinar`, `testar_sementes`,
`testar_variantes`) gravam três submissões por modelo. Para não encher `evaluation\results`, mande
as deles para outra pasta e compare só as que interessam, com nomes curtos (a tabela corta o nome
em 24 caracteres, e `..._nuno_suave` e `..._nuno_rigido` ficariam iguais):

```powershell
# antes de rodar os testes, na mesma janela: as submissões vão para evaluation\experimentos
$env:POSSE_ML_RESULTADOS = "$PWD\evaluation\experimentos"

# o que já estiver em results vai para lá também (move, não apaga)
New-Item -ItemType Directory -Force evaluation\experimentos | Out-Null
Move-Item evaluation\results\posse_ml_*.csv evaluation\experimentos -Force

# uma pasta só com as escolhidas: "nome curto" = o que vem depois de posse_ml_viterbi_
Remove-Item -Recurse -Force evaluation\comparacao -ErrorAction SilentlyContinue
New-Item -ItemType Directory evaluation\comparacao | Out-Null
$sel = @{ "tcn0_suave"="tcn_nuno_suave"; "tcn_sem_suave"="tcn_sementes_nuno_suave"; "hgb_suave"="nuno_suave" }
foreach ($k in $sel.Keys) { Copy-Item "evaluation\experimentos\posse_ml_viterbi_$($sel[$k]).csv" "evaluation\comparacao\$k.csv" }
py -3.13 evaluation/run_evaluation.py --results evaluation\comparacao --out evaluation\comparacao --no-detail --bootstrap 1000
```

`evaluation\experimentos` e `evaluation\comparacao` são cópias locais; não vão para o Git.

## 5. O que foi testado

### 5.1 Perfil da bola: 5 minutos contra o jogo inteiro

O primeiro perfil separou bola viva de morta pelo GT público. Nesses 5 minutos, todas as 38 lacunas
da bola do Viterbi com o jogo rolando têm menos de 1 s. Treinado assim, o modelo só vê lacuna longa
dentro de parada e aprende "lacuna longa = parada".

Teste que mostrou o problema: os modelos treinados com esse perfil, alimentados com a bola do ATD
(que tem lacunas de 3 s ou mais com o jogo rolando), marcaram como bola morta 517 dos 6.031 frames de
bola viva da janela, contra 146 com a fusão e 187 com o Viterbi, e o spellF1 caiu para 0,533.

No jogo inteiro, a bola do Viterbi tem 34 lacunas de 2 s ou mais com o jogo rolando (100 de 1 s ou
mais). Por isso o perfil passou a ser medido no jogo inteiro, com viva/morta pelo detector de paradas
(`medir_bola` usa o `paradas_D.csv` por padrão). Ao mesmo tempo, a degradação da bola parada passou a
seguir o histograma medido: a nossa bola some em pedaços curtos nas paradas (das 724 lacunas de bola
parada, só 12 passam de 6 s), e não na parada inteira.

| perfil (jogo inteiro) | presença | pulos | falta com bola viva | falta com bola morta |
|---|---|---|---|---|
| Viterbi | 83,5% | 0,6% | 8,9% | 32,0% |
| fusão | 72,8% | 3,4% | 18,0% | 51,4% |

### 5.2 As três bolas (`comparar_bolas`)

| | ATD | fusão | Viterbi |
|---|---|---|---|
| janela: bola presente, jogo rolando | 93,5% | 92,9% | 98,1% |
| janela: bola presente, bola parada | 66,4% | 79,9% | 87,9% |
| janela: o jogador mais perto é do time com a posse | 79,0% | 81,1% | 79,8% |
| janela: distância ao jogador mais perto do time com a posse | 1,89 m | 1,54 m | 1,58 m |
| jogo inteiro: bola presente | 59,3% | 70,4% | 83,1% |
| jogo inteiro: bola presente, jogo rolando | 77,7% | 82,0% | 91,1% |
| jogo inteiro: lacunas de 2 s ou mais, jogo rolando | 51 | 59 | 34 |
| pulos | 0,0% | 3,4% | 0,6% |

O Viterbi é a melhor bola nos dois lugares, mas a diferença é pequena na janela (a bola do ATD é
excepcionalmente boa nesses 5 minutos) e grande no resto do jogo. Por isso o placar da janela não
separa as bolas.

### 5.3 Validação na DFL (`treinar --rapido`)

| instância e perfil | spellF1 médio | macroF1 médio | custos escolhidos |
|---|---|---|---|
| Viterbi, perfil de 5 min | 0,610 | 0,914 | troca 6, parada 12, reinício 1 |
| Viterbi, perfil do jogo inteiro | 0,596 | 0,909 | troca 9, parada 12, reinício 1 |
| fusão, perfil do jogo inteiro | 0,580 | 0,904 | troca 9, parada 12, reinício 1 |

Com o perfil do jogo inteiro, a DFL é degradada com mais falhas, e a validação cai. Os números só
se comparam dentro do mesmo perfil: a fusão é treinada com uma bola bem pior (18% de falta com o
jogo rolando), e a validação mostra isso.

### 5.4 Bola morta (`comparar_paradas`, janela pública)

| fonte | acerto | precisão D | recall D | paradas (GT 4) | inventadas |
|---|---|---|---|---|---|
| detector de paradas (Nuno) | 97,4% | 88,8% | 99,4% | 5 | 1 |
| `posse_ml_fusao` | 96,6% | 90,9% | 91,7% | 4 | 0 |
| `posse_ml_viterbi` | 95,9% | 88,1% | 91,4% | 5 | 1 |

Bordas (fonte - GT, em frames): o detector começa as paradas sempre cedo (-25, -21, -14, -2) e
termina quase sempre tarde (-9, +15, +40, +59); as paradas dele são largas. O `posse_ml` termina um
pouco tarde (-7 a +25) e erra o começo de forma instável (-71, -8, +21, +94).

### 5.5 Janela pública (`run_evaluation`)

| versão | spellF1 | macroF1 | posses (GT 17) |
|---|---|---|---|
| Viterbi + bola morta do GT (**teto**, não vale) | 0,690 | 0,958 | 12 |
| Viterbi + detector, **suave** | **0,690** | **0,941** | 12 |
| Viterbi + detector, **rígido** | **0,690** | 0,931 | 12 |
| Viterbi, bola morta própria | 0,667 | 0,913 | 13 |
| fusão (própria, rígido, suave, teto) | 0,662 | 0,923 a 0,959 | 12 |
| Viterbi, perfil de 5 min | 0,645 | 0,910 | 14 |
| modelos do Viterbi (perfil de 5 min) com a bola do ATD | 0,533 | 0,861 | 13 |
| heurística `posse/` (`posse_v2`, bola morta do detector) | 0,586 | 0,920 | 11 |
| baseline oficial `smooth50` | 0,326 | 0,621 | 26 |

- **Teto:** o teste em que a bola morta vem do próprio GT, e o modelo só decide H ou A. Mostra o
  máximo que dá para ganhar melhorando a bola morta. Com o detector de paradas, o spellF1 já bate
  no teto nesta janela; melhorar a bola morta só mexe no macroF1.
- **Mesma bola morta, H/A diferente:** com a bola morta do detector nos dois, o H/A aprendido deu
  0,690 contra 0,586 da heurística do `posse/` (ainda dentro do ruído).
- **O que sobra está no H/A:** todas as versões preveem 12 posses contra 17, mesmo com a bola morta
  perfeita. Perdem ~5 trocas de posse com o jogo rolando. A matriz de confusão mostra um viés: ~250
  frames do Barça chamados de Athletic, contra ~95 no sentido contrário. (Isso vale para as árvores;
  a TCN, nas seções 5.6 e 5.8, acha as 17.)

### 5.6 Classificadores (`comparar_classificadores`, instância `viterbi`, perfil do jogo inteiro)

| classificador | treino | validação DFL (spellF1 / macroF1) | janela, bola morta própria | janela, detector rígido | janela, detector suave |
|---|---|---|---|---|---|
| `hgb` | — | 0,596 / 0,909 | 0,667 / 0,913 (13) | 0,690 / 0,931 (12) | 0,690 / 0,940 (12) |
| `xgboost` | 7 min (GPU) | 0,596 / 0,909 | 0,667 / 0,914 (13) | 0,690 / 0,931 (12) | 0,690 / 0,942 (12) |
| `lightgbm` | 8 min (CPU) | 0,591 / 0,909 | 0,645 / 0,906 (14) | 0,690 / 0,930 (12) | 0,690 / 0,940 (12) |
| `catboost` | 3 min (GPU) | 0,572 / 0,904 | 0,625 / 0,914 (15) | 0,690 / 0,933 (12) | 0,667 / 0,943 (13) |
| `gru` | ~5 min (GPU) | **0,613 / 0,916** | 0,613 / 0,913 (15) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) |
| `tcn` | 5 min (GPU) | 0,609 / 0,913 | 0,678 / 0,907 (19) | **0,727 / 0,933 (16)** | **0,729 / 0,948 (17)** |

Janela: spellF1 / macroF1 (posses previstas; GT 17). Validação com `--rapido` (150 árvores, ou 5
épocas nas redes).

- **As árvores empatam.** HistGradientBoosting, XGBoost e LightGBM ficam em 0,59–0,60 na DFL e
  quase iguais na janela; o CatBoost fica um pouco abaixo. Com as mesmas colunas por frame, trocar
  o tipo de árvore não muda nada.
- **As redes ganham na DFL.** A GRU é melhor que o XGBoost nos 7 jogos (+0,009 a +0,027 cada); a
  TCN, em 5 dos 7. Ler o trecho inteiro ajuda justamente nas trocas de posse com a bola rolando,
  que era onde as árvores perdiam.
- **Na janela, a TCN passa do antigo teto** (a semente 0; a seção 5.9 mostra que boa parte disso
  foi sorte). Com o detector de paradas, 0,727 (rígido) e 0,729
  (suave), com 16 e 17 posses contra 17 do GT. O teto de 0,690 da seção 5.5 era o das árvores, com
  a bola morta perfeita; a TCN acha mais trocas de posse com a bola viva. A GRU, que empata com a
  TCN na DFL, fica bem abaixo na janela: 17 posses são poucas para separar as duas.
- **O GRU "falhou"** só na saída do processo: no Windows, o PyTorch com CUDA às vezes derruba o
  Python depois de tudo gravado (código 3221226505). Os modelos estavam completos e entraram na
  comparação; o `comparar_classificadores` agora reconhece esse caso.

### 5.7 GRU + TCN (`combinar`)

A média das duas redes, em log-odds, com os custos do Viterbi escolhidos de novo na DFL (troca 9,
parada 12, reinício 1):

| jogo de fora | gru | tcn | gru+tcn |
|---|---|---|---|
| J03WMX | **0,551** | 0,534 | 0,545 |
| J03WN1 | **0,571** | 0,550 | 0,562 |
| J03WOH | 0,622 | **0,636** | 0,622 |
| J03WOY | 0,620 | **0,624** | 0,620 |
| J03WPY | 0,628 | **0,630** | 0,627 |
| J03WQQ | **0,637** | 0,629 | 0,633 |
| J03WR9 | 0,660 | 0,661 | **0,676** |
| **média DFL** | **0,613** / 0,916 | 0,609 / 0,913 | 0,612 / 0,915 |
| janela, própria | 0,613 / 0,913 (15) | **0,678** / 0,907 (19) | 0,612 / 0,908 (17) |
| janela, detector rígido | 0,676 / 0,930 (12) | **0,727** / 0,933 (16) | 0,658 / 0,935 (14) |
| janela, detector suave | 0,653 / 0,945 (13) | **0,729** / 0,948 (17) | 0,650 / 0,950 (15) |

**A média não ajudou.** Na DFL ela fica entre as duas e só passa a melhor em 1 dos 7 jogos; na
janela, se comporta como a GRU. As duas redes leem as mesmas colunas e treinam nos mesmos dados, e
erram nos mesmos lances: a média só ajuda quando os erros são diferentes. Fica descartada.

Na DFL, GRU e TCN empatam (0,004 de diferença; a TCN ganha 4 dos 7 jogos, por pouco, e a GRU os
outros 3, por mais). Só a janela separa as duas, e com folga.

### 5.8 Teto das redes (bola morta do GT na janela)

Com a semente 0 de cada rede; a seção 5.9 repete com outras sementes.

| | teto (bola morta do GT) | detector, rígido | detector, suave |
|---|---|---|---|
| `tcn` | **0,729 / 0,965 (17)** | 0,727 / 0,933 (16) | 0,729 / 0,948 (17) |
| `hgb` | 0,690 / 0,958 (12) | 0,690 / 0,931 (12) | 0,690 / 0,940 (12) |
| `gru` | 0,667 / 0,961 (13) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) |

`python -m posse_ml --classificador tcn --paradas evaluation\gt\BAR-ATH_possession_gt_public.csv`
(grava `posse_ml_viterbi_tcn_gt_rigido.csv`, que usa o GT: **nunca enviar**).

- **Com a TCN, a bola morta deixou de ser o gargalo do spellF1.** Com o detector, ela já está no
  teto (0,727–0,729 contra 0,729). Melhorar a bola morta só mexe no macroF1 (até 0,965), que é o
  desempate.
- **A diferença entre as redes está toda no H/A.** Com a mesma bola morta perfeita, a TCN acha 17
  posses, o hgb 12 e a GRU 13. Neste jogo, só a TCN pega as trocas de posse com a bola rolando que
  as outras perdem; a GRU fica abaixo até do hgb, embora seja melhor que ele na DFL.
- **O que falta para a TCN são as bordas.** Ela acerta o número de posses (17 de 17), mas o spellF1
  pede sobreposição alta com cada posse do GT: sobra o momento exato das trocas.
- A GRU com o detector (0,676) passou um pouco do próprio teto (0,667): as paradas do detector são
  mais largas e cortam as posses de outro jeito. Com tão poucas posses, é ruído.

### 5.9 Sementes (`testar_sementes`)

Três sementes de cada rede (0, 1 e 2), mudando só o sorteio da rede; as cópias da DFL são as
mesmas. `<rede>_sementes` é a média das três, validada na DFL com os custos escolhidos de novo.

| modelo | validação DFL | janela, própria | janela, detector rígido | janela, detector suave | teto (GT, não vale) |
|---|---|---|---|---|---|
| `tcn` (semente 0) | 0,609 / 0,913 | 0,678 / 0,907 (19) | 0,727 / 0,933 (16) | 0,729 / 0,948 (17) | 0,729 / 0,965 (17) |
| `tcn_s1` | 0,611 / 0,913 | 0,635 / 0,909 (17) | 0,671 / 0,935 (14) | 0,675 / 0,952 (15) | 0,675 / 0,967 (15) |
| `tcn_s2` | 0,611 / 0,914 | 0,630 / 0,902 (16) | 0,667 / 0,933 (13) | 0,693 / 0,947 (13) | 0,693 / 0,964 (13) |
| **`tcn_sementes`** | **0,616 / 0,915** | 0,635 / 0,911 (17) | 0,671 / 0,936 (14) | 0,675 / 0,951 (15) | 0,675 / 0,968 (15) |
| `gru` (semente 0) | 0,613 / 0,916 | 0,613 / 0,913 (15) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) | 0,667 / 0,961 (13) |
| `gru_s1` | 0,616 / 0,916 | 0,629 / 0,905 (18) | 0,679 / 0,924 (16) | 0,647 / 0,938 (17) | 0,671 / 0,955 (17) |
| `gru_s2` | 0,612 / 0,915 | 0,606 / 0,912 (16) | 0,640 / 0,932 (13) | 0,680 / 0,945 (13) | 0,680 / 0,963 (13) |
| `gru_sementes` | 0,614 / 0,917 | 0,625 / 0,913 (15) | 0,676 / 0,930 (12) | 0,667 / 0,944 (13) | 0,667 / 0,961 (13) |

- **O 0,729 da TCN foi, em boa parte, sorte da semente 0.** As outras duas dão 0,67–0,69 na janela
  (média das três: 0,699 na suave, 0,688 na rígida), perto dos 0,690 do hgb. As 17 posses também
  eram da semente 0: as outras acham 13 a 15.
- **A TCN continua um pouco acima da GRU na janela** (médias 0,699 contra 0,660 na suave; teto
  0,699 contra 0,672), mas as faixas quase se encostam. Com 17 posses, não dá para afirmar.
- **Na DFL, as sementes quase não variam:** 0,609–0,611 na TCN e 0,612–0,616 na GRU. Esse é o
  ruído de sorteio: uma variante só é melhor de verdade se passar a referência por mais de ~0,005.
- **As redes ganham das árvores na DFL em todas as sementes** (0,609–0,616 contra 0,572–0,596).
  Essa vantagem é firme; a vantagem sobre o hgb na janela, não.
- **A média das sementes da TCN é a primeira combinação que ajuda:** 0,616, acima de cada semente
  (+0,005 a +0,007) e da melhor delas em 4 dos 7 jogos. Média de sementes tira o acaso do sorteio;
  a de GRU com TCN não ajudou porque as duas erram nos mesmos lances. A da GRU fica igual às
  sementes (0,614).

### 5.10 Épocas e transformer (`testar_variantes`)

A `tcn` de referência e a `tcn_e10` têm o mesmo modelo final (10 épocas, semente 0); muda só a
validação, que escolhe os custos do Viterbi.

| | épocas (validação / final) | custo de troca | DFL | janela, rígido | janela, suave | teto | posses (suave) | morta/viva por frame, só o modelo |
|---|---|---|---|---|---|---|---|---|
| `tcn` | 5 / 10 | 6 | 0,609 / 0,913 | 0,727 / 0,933 | 0,729 / 0,948 | 0,729 | 17 | 0,935 |
| `tcn_e10` | 10 / 10 | 9 | 0,633 / 0,918 | 0,671 / 0,936 | 0,675 / 0,951 | 0,675 | 15 | 0,935 |
| `tcn_e20` | 20 / 20 | 6 | 0,658 / 0,926 | 0,619 / 0,924 | 0,613 / 0,939 | 0,625 | 15 | 0,919 |
| `tcn_e30` | 30 / 30 | 6 | **0,661 / 0,926** | 0,594 / 0,919 | 0,613 / 0,934 | 0,613 | 15 | 0,909 |
| `lstm` | 5 / 10 | 6 | 0,619 / 0,915 | **0,727** / 0,931 | 0,715 / 0,944 | 0,715 | 16 | 0,928 |
| `transformer` | 5 / 10 | 6 | 0,595 / 0,910 | 0,700 / **0,944** | 0,713 / **0,955** | 0,725 | 15 | 0,916 |

- **A validação rápida subestimava as redes.** A mesma rede final dá 0,609 validada com 5 épocas e
  0,633 com 10: os números de DFL das redes nas seções 5.6 a 5.9 estão ~0,025 abaixo do modelo de
  verdade (entre as redes a comparação é justa; contra as árvores, a distância é maior do que
  parecia).
- **O custo de troca pesa muito na janela.** Com a mesma rede, troca 6 dá 17 posses e 0,729; troca
  9, 15 posses e 0,675. Todos os modelos preveem posses de menos (12 a 16 contra 17 na janela, 70 a
  80% na DFL), e a regra do platô escolhe de propósito a troca mais cara.
- **Mais épocas: a DFL sobe, o nosso jogo cai.** 20 épocas ganha de 10 nos 7 jogos da DFL (+0,017 a
  +0,036); 30 não acrescenta. Na janela, 20 e 30 caem para ~0,61, o teto cai junto, e o acerto de
  bola morta por frame, que é bem menos ruidoso que o spellF1, cai a cada época a mais (0,935, 0,919,
  0,909). A explicação provável: treinando mais, a rede aprende detalhes da DFL (liga, rótulo,
  degradação imitada) que não valem no nosso jogo; a validação também é DFL e premia isso. Parar
  cedo protege. Para épocas, a DFL não decide sozinha.
- **O transformer é o mais fraco das redes na DFL e vai bem no nosso jogo.** Na DFL fica em 0,595,
  0,015 a 0,025 abaixo das outras redes e empatado com as árvores (transformer precisa de mais dado
  que 7 jogos). Na janela, 0,700 a 0,713, perto da TCN e da LSTM, com o melhor macroF1 de todos
  (0,955 na suave, 0,974 no teto) e as bordas de começo de parada mais perto do GT. É a segunda vez
  que a DFL e o nosso jogo discordam: uma leitura possível é que o modelo que se ajusta menos à DFL
  transfere melhor. Uma semente só.
- **A LSTM é a única sem conflito entre as duas réguas:** a melhor das redes validadas com 5 épocas
  na DFL (0,619) e 0,715 a 0,727 na janela. Uma semente só.
- As variantes `tcn_e5`, `lstm_e5` e `lstm_e20` existem no `testar_variantes`, mas o teste que vale
  para épocas passou a ser o `testar_epocas` (seção 6), com 3 sementes por configuração.

### 5.11 Épocas com sementes novas (`testar_epocas --sementes 10 11 12`)

LSTM, transformer e TCN com 5, 10 e 20 épocas no modelo final, 3 sementes novas cada (27 modelos),
custos do Viterbi sempre da validação de 5 épocas. Média das 3 sementes (e o valor de cada uma):

| rede | épocas | DFL | janela, rígido | janela, suave | teto | morta/viva por frame | H/A por frame |
|---|---|---|---|---|---|---|---|
| **tcn** | **5** | **0,618** | **0,728** (0,729 / 0,727 / 0,727) | **0,733** | **0,749** | 0,933 | 0,953 |
| tcn | 10 | 0,618 | 0,698 (0,671 / 0,684 / 0,739) | 0,693 | 0,709 | 0,920 | 0,952 |
| tcn | 20 | 0,618 | 0,636 (0,606 / 0,645 / 0,658) | 0,645 | 0,645 | 0,912 | 0,948 |
| lstm | 5 | 0,615 | 0,717 (0,741 / 0,706 / 0,703) | 0,718 | 0,737 | 0,934 | 0,953 |
| lstm | 10 | 0,615 | 0,695 (0,707 / 0,693 / 0,684) | 0,709 | 0,714 | 0,923 | 0,951 |
| lstm | 20 | 0,615 | 0,691 (0,676 / 0,710 / 0,688) | 0,710 | 0,714 | 0,931 | 0,944 |
| transformer | 5 | 0,593 | 0,658 (0,688 / 0,637 / 0,648) | 0,674 | 0,680 | 0,919 | 0,954 |
| transformer | 10 | 0,593 | 0,687 (0,688 / 0,671 / 0,703) | 0,698 | 0,703 | 0,926 | 0,955 |
| transformer | 20 | 0,593 | 0,684 (0,693 / 0,671 / 0,688) | 0,699 | 0,710 | 0,930 | 0,950 |

A DFL é a mesma para as três épocas da mesma rede (a validação é sempre de 5 épocas); quem compara
as épocas é a janela. Tabela completa, modelo a modelo, em `modelos/viterbi/epocas.md`.

Regra combinada antes de ver o resultado: média do spellF1 no rígido; diferença menor que 0,03 é
empate, desempatado pelo macroF1; as medidas por frame conferem.

- **TCN com 5 épocas é a campeã.** 0,728 no rígido contra 0,717 da LSTM com 5: pela regra, empate
  técnico, e o macroF1 também empata (0,937 e 0,937). Mas a TCN-5 fica à frente no suave, no teto, na
  bola morta própria e na DFL, e é de longe a mais estável: 0,729 / 0,727 / 0,727 nas três sementes,
  todas com 16 ou 17 posses (GT 17), contra 0,703 a 0,741 da LSTM-5.
- **Treinar menos é melhor no nosso jogo para TCN e LSTM.** A TCN cai a cada época a mais (0,728,
  0,698, 0,636), com o acerto de bola morta por frame caindo junto (0,933, 0,920, 0,912), como na
  seção 5.10. A LSTM tem o melhor resultado com 5 e fica estável depois.
- **O transformer é a exceção:** com 5 épocas ainda não aprendeu o bastante (0,658) e melhora com 10
  (0,687); aprende mais devagar, como a pesquisa dizia. Fica abaixo das outras duas.
- **Com 5 épocas, DFL e janela concordam** (TCN ≥ LSTM > transformer nas duas). É também a única
  configuração em que a rede validada é exatamente a rede enviada, e os custos do Viterbi foram
  escolhidos para ela.
- **O 0,729 da antiga TCN semente 0 não era só sorte:** é o nível típico da TCN de 5 épocas; aquela
  rede tinha 10 épocas, mas os custos vinham da validação de 5.

### 5.12 Árvores com menos árvores e mais rasas (`testar_arvores --classificador xgboost`)

Validação com o mesmo número de árvores do modelo final; árvore não depende de semente.

| XGBoost (árvores / folhas) | DFL | janela, rígido | janela, suave | teto | posses | morta/viva por frame | H/A por frame |
|---|---|---|---|---|---|---|---|
| 50 / 15 | 0,538 | 0,662 | 0,653 | 0,645 | 12 | 0,910 | 0,940 |
| 150 / 15 | 0,574 | 0,662 | 0,662 | 0,680 | 12 | 0,913 | 0,942 |
| 400 / 15 | 0,596 | 0,676 | 0,676 | 0,676 | 12 | 0,917 | 0,939 |
| 50 / 63 | 0,562 | 0,648 | 0,648 | 0,648 | 12 | 0,902 | 0,946 |
| 150 / 63 | 0,596 | 0,662 | 0,662 | 0,662 | 12 | 0,914 | 0,943 |
| **400 / 63** | **0,603** | **0,690** | **0,690** | **0,690** | 12 | 0,915 | 0,943 |
| referência (400 / 63, validada com 150) | 0,596 | 0,690 | 0,690 | 0,690 | 12 | 0,915 | 0,943 |

- **Nas árvores, treinar menos piora as duas réguas.** As redes se especializavam demais na DFL; as
  árvores aprendem de menos (a taxa 0,05 e o mínimo por folha já seguram bastante). A configuração
  padrão continua a melhor.
- **O limite é a família:** toda árvore prevê 12 posses na janela e nenhuma passa de 0,690, nem no
  teto. Por frame, o H/A das árvores (~0,94) é quase o das redes (~0,95); o que as redes fazem a mais
  é encadear os frames e achar as trocas de posse.
- Validar com as mesmas 400 árvores deu 0,603 na DFL, contra 0,596 com 150; na janela, igual.

### 5.13 Menos épocas (`testar_epocas --validacao igual`, sementes 10, 11 e 12)

Validação com as mesmas épocas do modelo final. Média das 3 sementes:

| rede | épocas | DFL | janela, rígido | janela, suave | teto | morta/viva por frame | H/A por frame |
|---|---|---|---|---|---|---|---|
| tcn | 1 | 0,536 | 0,648 | 0,648 | 0,648 | 0,938 | 0,944 |
| tcn | 2 | 0,585 | 0,644 | 0,655 | 0,664 | 0,937 | 0,947 |
| tcn | 3 | 0,600 | 0,689 | 0,688 | 0,711 | 0,939 | 0,950 |
| **tcn** | **5** | **0,618** | **0,728** | **0,733** | **0,749** | 0,933 | 0,953 |
| lstm | 1 | 0,519 | 0,636 | 0,645 | 0,650 | 0,921 | 0,942 |
| lstm | 2 | 0,583 | 0,654 | 0,667 | 0,675 | 0,933 | 0,947 |
| lstm | 3 | 0,599 | 0,675 | 0,685 | 0,693 | 0,934 | 0,949 |
| lstm | 5 | 0,615 | 0,717 | 0,718 | 0,737 | 0,934 | 0,953 |

Tabela completa, modelo a modelo, em `modelos/viterbi/epocas_igual.md`.

- **Menos de 5 épocas piora as duas réguas,** nas duas redes: a rede ainda não aprendeu o bastante
  (com 1 época, as três sementes da TCN dão exatamente 0,648, 12 posses: quase não troca de posse).
- **Juntando com as seções 5.10 e 5.11, 5 épocas é o ponto certo:** na DFL a curva sobe até 20–30
  épocas; na janela ela sobe até 5 e cai depois (TCN, rígido: 0,648, 0,644, 0,689, **0,728**, ~0,70
  com 10, ~0,63 com 20). As duas réguas se cruzam em 5.

### 5.14 Variação entre sementes (desvio padrão das 3 sementes)

Média ± desvio padrão amostral das sementes 10, 11 e 12 (seções 5.11 e 5.13). Com 1, 2, 3 e 5
épocas, a validação usa as mesmas épocas do modelo final; com 10 e 20, a validação é de 5 épocas.
Janela: spellF1.

| rede | épocas | DFL | janela, rígido | janela, suave | teto |
|---|---|---|---|---|---|
| tcn | 1 | 0,536 ± 0,003 | 0,648 ± 0,000 | 0,648 ± 0,000 | 0,648 ± 0,000 |
| tcn | 2 | 0,585 ± 0,006 | 0,644 ± 0,016 | 0,655 ± 0,003 | 0,664 ± 0,005 |
| tcn | 3 | 0,600 ± 0,007 | 0,689 ± 0,027 | 0,687 ± 0,028 | 0,711 ± 0,034 |
| **tcn** | **5** | 0,618 ± 0,002 | **0,728 ± 0,001** | **0,733 ± 0,018** | **0,749 ± 0,014** |
| tcn | 10 | 0,618 ± 0,002 | 0,698 ± 0,036 | 0,694 ± 0,022 | 0,710 ± 0,038 |
| tcn | 20 | 0,618 ± 0,002 | 0,636 ± 0,027 | 0,645 ± 0,019 | 0,645 ± 0,019 |
| lstm | 1 | 0,519 ± 0,005 | 0,636 ± 0,003 | 0,645 ± 0,019 | 0,649 ± 0,027 |
| lstm | 2 | 0,583 ± 0,004 | 0,654 ± 0,040 | 0,667 ± 0,041 | 0,675 ± 0,046 |
| lstm | 3 | 0,599 ± 0,005 | 0,675 ± 0,035 | 0,685 ± 0,032 | 0,692 ± 0,041 |
| **lstm** | **5** | 0,615 ± 0,002 | **0,717 ± 0,021** | **0,718 ± 0,036** | **0,737 ± 0,036** |
| lstm | 10 | 0,615 ± 0,002 | 0,695 ± 0,012 | 0,709 ± 0,030 | 0,714 ± 0,023 |
| lstm | 20 | 0,615 ± 0,002 | 0,691 ± 0,017 | 0,710 ± 0,026 | 0,714 ± 0,030 |

(Com 10 e 20 épocas, a DFL é a da validação de 5 épocas, a mesma das redes de 5.)

- **Na janela, uma semente oscila em torno de 0,02 a 0,04** (desvio típico; ~1 posse das 17). É
  por isso que a regra de escolha trata diferenças menores que 0,03 como empate. Na DFL, a
  oscilação é bem menor (0,002 a 0,007).
- **A TCN-5 no rígido é excepcionalmente estável** (0,729 / 0,727 / 0,727, desvio 0,001). No suave
  e no teto o desvio dela volta ao normal (0,014 a 0,018), então parte dessa estabilidade é acaso
  de ter só 3 sementes. A LSTM-5 oscila como as outras configurações (0,021 no rígido).
- **O pico em 5 épocas é moderado em cada rede e firme no conjunto.** A diferença da TCN-5 para as
  vizinhas (3 e 10 épocas) é de 0,03 a 0,04, uma a duas vezes o desvio entre sementes; com 3
  sementes, o erro padrão de cada média é ~desvio/√3 ≈ 0,01 a 0,02. Sozinha, cada comparação seria
  fraca; o que dá força é o mesmo pico aparecer nas duas redes, no rígido, no suave e no teto, e
  bater com a TCN de 10 épocas das sementes 0, 1 e 2 (média 0,688, seção 5.9).
- **Cuidado com o próprio desvio:** com 3 valores, ele é impreciso (pode errar pela metade para mais
  ou para menos). Serve para a ordem de grandeza, não para comparar desvios entre si.

### 5.15 Outras configurações (`testar_configs --epocas 5`, sementes 10, 11 e 12)

Cada configuração muda uma coisa em relação à padrão (largura 64, dropout 0,1, TCN com 6 blocos,
LSTM com 2 camadas); 5 épocas na validação e no modelo final. Ordenado pelo spellF1 no rígido, média
± desvio padrão das 3 sementes:

| # | configuração | janela, rígido | pior semente | janela, suave | teto | DFL |
|---|---|---|---|---|---|---|
| 1 | **TCN padrão** | **0,728 ± 0,001** | 0,727 | 0,733 | 0,749 | 0,618 |
| 2 | TCN, dropout 0,3 | 0,720 ± 0,006 | 0,715 | 0,714 | 0,733 | 0,615 |
| 3 | **LSTM padrão** | 0,717 ± 0,021 | 0,703 | 0,718 | 0,737 | 0,615 |
| 4 | LSTM, 1 camada | 0,717 ± 0,020 | 0,694 | 0,718 | 0,733 | 0,615 |
| 5 | LSTM, largura 128 | 0,716 ± 0,013 | 0,703 | 0,702 | 0,718 | 0,624 |
| 6 | TCN, largura 128 | 0,708 ± 0,044 | 0,658 | 0,703 | 0,711 | 0,626 |
| 7 | LSTM, 3 camadas | 0,697 ± 0,043 | 0,648 | 0,695 | 0,703 | 0,612 |
| 8 | LSTM, dropout 0,3 | 0,688 ± 0,030 | 0,658 | 0,700 | 0,704 | 0,614 |
| 9 | TCN, 7 blocos (~±20 s) | 0,685 ± 0,024 | 0,662 | 0,690 | 0,700 | 0,599 |
| 10 | TCN, largura 32 | 0,684 ± 0,030 | 0,662 | 0,685 | 0,688 | 0,604 |
| 11 | LSTM, largura 32 | 0,671 ± 0,023 | 0,648 | 0,675 | 0,676 | 0,606 |
| 12 | TCN, 4 blocos (~±2,5 s) | 0,616 ± 0,012 | 0,606 | 0,616 | 0,627 | 0,607 |

O macroF1 no rígido fica entre 0,935 e 0,940 em todas. Tabela completa em `modelos/viterbi/configs.md`.

- **Nenhuma configuração passa a padrão.** As cinco primeiras estão dentro de 0,012 (empate pela
  regra dos 0,03); em empate fica a padrão, a mais simples e a mais testada.
- **Largura 128 melhora a DFL e não o nosso jogo** (e deixa a TCN instável entre sementes): o mesmo
  padrão das épocas, mais capacidade ajuda na DFL e não transfere.
- **O alcance da TCN importa:** ±2,5 s é o pior resultado de todos (0,616), ±20 s também piora;
  ±10 s é o ponto certo. Redes menores (largura 32) pioram nas duas réguas.

## 6. Estado atual e próximos passos

**Candidatas: TCN e LSTM, as duas com 5 épocas** (validação e modelo final com 5 épocas), com o
detector de paradas. O transformer está descartado.

| candidata | modelos | janela, rígido (média das 3 sementes) | janela, suave | DFL |
|---|---|---|---|---|
| **TCN, 5 épocas** (à frente) | `tcn_f5_s10`, `tcn_f5_s11`, `tcn_f5_s12` | **0,728** (0,729 / 0,727 / 0,727) / 0,937 | **0,733** / 0,953 | **0,618** |
| LSTM, 5 épocas | `lstm_f5_s10`, `lstm_f5_s11`, `lstm_f5_s12` | 0,717 (0,741 / 0,706 / 0,703) / 0,937 | 0,718 / 0,953 | 0,615 |

Pela regra combinada (seção 5.11), as duas empatam: diferença de 0,011 no rígido, menor que 0,03, e
o mesmo macroF1. A TCN fica à frente em todas as outras colunas e é bem mais estável entre sementes.

Para gerar a submissão de cada uma, a média das três sementes:

```powershell
$env:POSSE_ML_RESULTADOS = "$PWD\evaluation\experimentos"
py -3.13 -m posse_ml.combinar --classificadores tcn_f5_s10 tcn_f5_s11 tcn_f5_s12
py -3.13 -m posse_ml.combinar --classificadores lstm_f5_s10 lstm_f5_s11 lstm_f5_s12
```

Gravam `posse_ml_viterbi_<rede>_f5_s10+<rede>_f5_s11+<rede>_f5_s12_nuno_{rigido,suave}.csv` (e a
própria). Falta escolher entre rígido (nada ajustado na janela) e suave (pesos ajustados na janela),
o que depende de saber como o detector de paradas foi ajustado.

Descartados, e por quê:

| | janela, rígido | motivo |
|---|---|---|
| transformer (5, 10 e 20 épocas) | 0,658 a 0,687 | abaixo da TCN e da LSTM nas duas réguas; precisa de mais dados que os 7 jogos da DFL |
| TCN e LSTM com 10 ou 20 épocas | 0,636 a 0,698 | treinar mais piora no nosso jogo (seções 5.10 e 5.11) |
| TCN semente 0 de 10 épocas (`tcn`) e a média das TCN de 10 épocas | 0,727 e 0,671 | 10 épocas; a TCN-5 dá o mesmo nível com as três sementes |
| GRU | 0,640 a 0,679 | abaixo da LSTM, que é a mesma ideia |
| árvores (HGB, XGBoost, LightGBM, CatBoost) | 0,690 | as redes ganham delas nas duas réguas |

### Próximos passos

Feitos: menos árvores (5.12), menos épocas (5.13), outras configurações (5.15). A configuração
padrão com 5 épocas fica, na TCN e na LSTM. A validação é pela janela pública (os 5 minutos do GT):
rotular mais GT à mão saiu do plano.

| quando | o quê |
|---|---|
| qua 11/11 | **dados de treino** (`testar_dados`: só DFL, só Metrica, DFL + Metrica) e **confirmação em sementes novas** (`testar_epocas --classificadores tcn lstm --epocas 5 --sementes 10 11 12 20 21 22 --validacao igual`) |
| qui 12/11 | fechar os dados de treino; **dois modelos finais** (TCN e LSTM), cada um a média das sementes (`combinar`); CSVs em `evaluation/results` e pesos no GitHub para o time |
| sex 13/11 | **auditoria das regras** e teste de reprodução num clone limpo |
| sáb 14 – dom 15/11 | **documento técnico** e limpeza do repositório |
| seg 16/11 | revisão final, escolha do CSV (um por equipe) e **envio** |
| ter 17/11 | folga (prazo) |

Notas para esses passos:

- **Épocas e dados de treino:** uma época é uma passada pelos dados; com só a Metrica (3 jogos), 5
  épocas são menos da metade das atualizações de rede que com a DFL (7 jogos), e com DFL + Metrica,
  ~40% a mais. Como o resultado é sensível às épocas, comparar também com o número de atualizações
  igualado (~12 épocas para só Metrica, 3 ou 4 para DFL + Metrica).
- **Modelo final = média das sementes, não uma semente.** Entre sementes da mesma configuração, a
  qualidade real quase não muda (0,002 a 0,007 na DFL) e a sorte na janela muda 0,02 a 0,04:
  escolher a semente campeã da janela é escolher pela sorte (a primeira TCN fez 0,729 e as duas
  seguintes, 0,67). A média mantém o sinal comum e dilui os erros de cada rede.
- **Rígido ou suave:** o detector de paradas foi desenvolvido olhando os 4 trechos de bola morta da
  janela (sem otimização direta nela), e o suave ainda ajusta 4 pesos na janela; o rígido é a
  escolha mais segura.
- **Auditoria:** uso do GT público na calibração (pesos do suave; limite de 2 s dos rótulos da
  Metrica), declaração dos dados externos (DFL, Metrica), o detector de paradas, repositório
  público e reprodutível com os modelos finais, nomes dos arquivos.
- **Posses de menos na DFL** (70 a 80% das reais): provavelmente é diferença de convenção (a DFL
  tem ~5 posses por minuto; o GT, 3,4), não erro. Fica só como observação.

Prazo do desafio: 17/11/2026.

## 7. Arquivos

```
posse_ml/
├─ config.py        caminhos, códigos dos rótulos, kickoffs
├─ janelas.py       médias em janelas, "desde" e "até", trechos
├─ partida.py       leitura do ATD e da nossa bola, correção dos goleiros
├─ bola.py          filtro de pulos, ponte, velocidade
├─ atributos.py     as 100 colunas por frame
├─ dfl.py           leitor dos jogos da DFL (só biblioteca padrão)
├─ perfil.py        perfil da bola: medida e sorteio dos histogramas
├─ degradacao.py    DFL -> cara do nosso jogo
├─ modelos.py       os classificadores de árvore (hgb, xgboost, lightgbm, catboost), salvar e carregar
├─ sequencia.py     as redes de sequência (gru, lstm, tcn)
├─ viterbi.py       decodificador de 4 estados, kickoff, adiantamento das paradas
├─ transmissao.py   alinhamento e plano da TV
├─ prever.py        do jogo aos rótulos
├─ __main__.py      python -m posse_ml: escreve a submissão
├─ baixar_dfl.py, medir_bola.py, treinar.py, calibrar.py, comparar_bolas.py, comparar_paradas.py,
│  comparar_classificadores.py, combinar.py, testar_sementes.py,
│  testar_variantes.py, testar_epocas.py, testar_arvores.py,
│  testar_configs.py, testar_dados.py
├─ metrica.py, baixar_metrica.py   os 3 jogos abertos da Metrica e os rótulos derivados dos eventos
├─ modelos/<instância>/  perfil_bola.json, decodificador.json, calibracao.json, conjuntos/*/ (vão para o Git)
│                   e uma subpasta por classificador além do hgb (modelos/viterbi/xgboost/, ...)
└─ testes/          testes sem dados (unittest) e um ponta a ponta sintético
```

**O que vai para o Git:** o código e `modelos/` (pickles dos classificadores de árvore, pesos das
redes em `rede.pt`, custos e pesos de calibração, e o perfil da bola, que só tem frações e
histogramas). **O que não vai:** `cache/` (o ATD em outro
formato, as cópias degradadas, as probabilidades) e `dfl/` (`posse_ml/.gitignore`). A bola do
tracker continua em `data/`, fora do Git.

Nenhum script envia dados do jogo para fora da máquina. O único acesso à rede é o `baixar_dfl.py`,
que só baixa os dados abertos da DFL.

## 8. Testes

```powershell
python -m unittest discover -s posse_ml/testes -t .   # segundos, sem dados
python -m posse_ml.testes.ponta_a_ponta                # ~5 min: o pipeline inteiro com dados sintéticos
python -m posse_ml.testes.ponta_a_ponta --classificadores xgboost lightgbm catboost gru tcn   # + comparar, combinar, sementes e variantes, ~15 min
```

Os testes dos outros classificadores pulam sozinhos os pacotes que não estiverem instalados.

O ponta a ponta monta numa pasta temporária um jogo sintético com os 199.319 frames, ATD, bola do
tracker, homografia, transmissão com cortes nas paradas e três "jogos da DFL", e roda
`medir_bola`, `treinar`, `calibrar` e a submissão. Os números dele não dizem nada sobre o jogo real:
só mostram que tudo encaixa e que a submissão sai válida.

## 9. Dados externos

Sportec Open DFL: Bassek, M., Rein, R., Weber, H. e Memmert, D. (2025). An integrated dataset of
spatiotemporal and event data in elite soccer. *Scientific Data* 12:195.
doi:10.1038/s41597-025-04505-y. Dados: doi:10.6084/m9.figshare.28196177. CC-BY 4.0.
