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

**Testar variantes das redes** (o número de épocas da TCN, a LSTM e o transformer):

```powershell
python -m posse_ml.testar_variantes                       # lstm, tcn_e10, tcn_e20, tcn_e30, transformer
python -m posse_ml.testar_variantes --variantes tcn_e20 lstm
python -m posse_ml.testar_variantes --so-comparar
```

| variante | o que muda | épocas (validação / final) |
|---|---|---|
| `lstm` | LSTM no lugar da GRU, mesma configuração | 5 / 10 |
| `tcn_e10` | nada no modelo final; a validação usa as mesmas 10 épocas (a `tcn` padrão valida com 5) | 10 / 10 |
| `tcn_e20` | 20 épocas | 20 / 20 |
| `tcn_e30` | 30 épocas | 30 / 30 |
| `transformer` | o transformer, configuração padrão | 5 / 10 |

A `tcn` e a `gru` que já existem entram como referência, sem treinar de novo. Saem a validação na
DFL, as três versões na janela e o teto, e a tabela fica em `modelos/<instância>/variantes.md`. Uma
variante só vale se passar a referência na DFL por mais do que a diferença entre sementes. Na GPU,
~30 a 40 minutos para as quatro da TCN e a LSTM, mais uns 5 para o transformer.

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

## 6. Estado atual e próximos passos

Três candidatas, todas da instância `viterbi` com a bola morta do detector de paradas, cada uma por
um motivo:

| candidata | por que | janela, suave | janela, rígido | DFL |
|---|---|---|---|---|
| TCN semente 0 (`tcn`) | a melhor na janela, o único pedaço do jogo de verdade | 0,729 / 0,948 | 0,727 / 0,933 | 0,609 |
| média das 3 TCN (`tcn_sementes`) | a melhor na DFL e a mais estável | 0,675 / 0,951 | 0,671 / 0,936 | **0,616** |
| HistGradientBoosting (`hgb`) | a referência das árvores, a mais simples | 0,690 / 0,940 | 0,690 / 0,931 | 0,596 |

Arquivos: `posse_ml_viterbi_tcn_nuno_{suave,rigido}.csv`, `posse_ml_viterbi_tcn_sementes_nuno_{suave,rigido}.csv`
e `posse_ml_viterbi_nuno_{suave,rigido}.csv` (em `evaluation\experimentos` se a pasta de resultados
foi limpa, seção 4).

Entre elas, as evidências puxam para lados diferentes:

- **A janela** é o jogo de verdade, mas tem 17 posses: as diferenças entre as três (0,675 a 0,729)
  são de 1 ou 2 posses. As outras sementes da TCN deram 0,675 e 0,693, o que mostra o quanto a
  janela oscila só pelo sorteio da rede (seção 5.9).
- **A DFL** tem 7 jogos inteiros, mas é uma imitação degradada do nosso tracker. Nela, todas as
  redes ganham de todas as árvores, e a média das sementes ganha de cada semente.
- Pela DFL, a aposta é a média das sementes; pela janela, a TCN semente 0. O que resolveria é mais
  GT do nosso jogo: rotular à mão mais 10 a 15 minutos do vídeo (só para avaliar, nunca para
  treinar) daria 30 a 50 posses a mais para comparar as três.

Pendências:

1. **Variantes das redes** (`python -m posse_ml.testar_variantes`): o transformer, a LSTM e a TCN com
   mais épocas (a perda ainda caía na época 10), todas com a semente 0. Comparar com a `tcn_e10` na
   DFL e com o ruído das sementes (~0,005). Se uma variante ganhar, treinar sementes dela e usar a
   média (o `testar_sementes` já aceita `--classificadores transformer`; as variantes de épocas da
   TCN ainda não). Profundidade, largura e alcance ficam para depois.
2. **Mais GT do nosso jogo**, rotulado à mão, para decidir entre as três candidatas.
3. **Bordas das trocas.** Abrir no viewer o GT e a candidata e ver se as trocas saem sempre cedo ou
   tarde.
4. **Saber se o detector de paradas foi ajustado olhando a janela pública.** Se foi, os números dele
   na janela estão otimistas, e o rígido passa a ser a escolha mais segura.
5. **Validação completa sem `--rapido`**, uma vez, para conferir se os custos escolhidos mudam.

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
│  testar_variantes.py
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
