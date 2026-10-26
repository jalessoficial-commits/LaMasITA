# Diagnóstico do tracking (ATD)

Esta pasta reúne os scripts que usamos para entender a qualidade do **ATD** (tracking automático
do vídeo tactical) antes de construir o modelo de posse. O ATD é a principal entrada do problema e
não foi corrigido à mão, então era preciso saber onde ele falha.

Nenhum script desta pasta lê o GT de posse: tudo sai só do ATD, da homografia e do vídeo.
Seguimos os itens que o `SUBMISSION.md` pede para o documento técnico: abordagem, assets usados,
limitações e reprodução.

| script | o que faz |
|---|---|
| `diagnostico.py` | estatísticas gerais do ATD; gera o cache usado pelos outros três |
| `assistir_slot.py` | toca o vídeo tactical com o tracking desenhado por cima |
| `listar_contagem.py` | lista os trechos em que um time tem slots demais ou de menos |
| `verificar_goleiro.py` | identifica quais slots se comportam como goleiro |

---

## 1. Abordagem

O ATD tem um registro por frame de vídeo, com 46 **slots** de jogador (`x, y` normalizados 0–1 e
velocidade) e a bola (`x, y`). Um slot não é um jogador: é um *tracklet*, que aparece, some e pode
trocar de identidade. O time de cada slot é **fixo** e vem do XML de metadados.

Queríamos responder três perguntas antes de usar o ATD como feature:
1. **A bola:** quanto falta e com que padrão (lacunas curtas ou longas)?
2. **Os slots:** com que frequência trocam de identidade, duplicam ou ficam com o time errado?
3. **Os lados:** os times trocam de lado no 2º tempo?

Os números dos scripts foram conferidos contra os publicados em `DATA.md`. Para as dúvidas que os
números não resolvem, assistimos ao vídeo com o tracking projetado pela homografia.

---

## 2. Passo a passo e resultados

Todos os comandos rodam a partir da raiz do repositório.

**Frames graduados** são os que o scorer avalia: frames 11.185–90.651 (1º tempo) e
110.808–190.197 (2º tempo), 158.857 no total.

### Passo 1 — Diagnóstico geral (`diagnostico.py`)

```bash
python diagnostico/diagnostico.py
```

Lê o `atd.txt` uma vez (1–2 min) e guarda o cache `data/_cache/atd_xy_v2.npz`. Depois imprime as
seções 0 a 8.

**Seção 0, sanidade do parse.** A cobertura da bola bate exatamente com o `DATA.md`: 113.488 de
199.319 frames no total e 94.214 de 158.857 nos graduados (59,3%). Isso valida o parse e o
alinhamento `video_frame = frame_id − 1`.

**Seção 1, slots por time** (frames graduados):

| | slots no XML | média por frame | 11 slots | mais de 11 | menos de 9 |
|---|---|---|---|---|---|
| casa (Barcelona) | 24 | 10,20 | 54,2% | 8,1% | 9,6% |
| fora (Athletic) | 22 | 10,29 | 61,0% | 4,7% | 8,1% |

O total médio é 20,49 objetos por frame (o `DATA.md` diz 20,5), e passa de 22 em 2,4% dos frames.
Em 1,0% dos frames graduados a casa tem só 2 slots. A causa disso não foi investigada.

**Seção 2, objetos por frame em cada segmento do vídeo:**

| segmento | frames | objetos/frame | bola presente |
|---|---|---|---|
| antes do jogo | 11.185 | 12,8 | 1,6% |
| 1º tempo (graduado) | 79.467 | 20,1 | 59,3% |
| intervalo | 20.156 | 12,5 | 94,1% |
| 2º tempo antes do apito (graduado, todo `D`) | 2.934 | 18,6 | 80,8% |
| 2º tempo em jogo (graduado) | 76.456 | 20,9 | 58,4% |
| depois do jogo | 9.121 | 14,5 | 1,5% |

Nenhum segmento tem frames sem objetos. O intervalo **não** é "quase vazio", como sugeria o
docstring do `starter/load_atd.py`. A bola aparece mais no intervalo e no trecho antes do apito do
que em jogo. **Bola presente no ATD não significa bola em jogo.**

**Seção 3, validade dos valores.** `NaN` é a única forma de ausência: não há x/y parcial nem
(0,0). Valores fora de [0,1] aparecem em 0,93% das posições de jogadores e em 4,19% das de bola
nos graduados. A bola chega a y = 1,34, ou seja, ~23 m fora do campo, o que provavelmente é
detecção espúria.

**Seção 4, saltos e reaparições** (frames graduados):
- **salto:** o slot se desloca mais de 2 m entre frames seguidos, isto é, mais de 50 m/s. Houve
  83 saltos, **todos do slot 24**;
- **reaparição impossível:** o slot volta de uma lacuna mais longe do que 10 m/s mais 2 m de
  folga permitem. Foram 23 de 1.139 reaparições (2,0%);
- cada slot tem, em média, 27,5 fragmentos.

Trocas de identidade por "teleporte" são raras. Elas se concentram num único slot.

**Seção 5, duração das lacunas** (frames graduados):

| | lacunas | mediana | p90 | máximo | frames ausentes em lacunas de mais de 10 s |
|---|---|---|---|---|---|
| jogadores (internas a cada slot) | 1.139 | 7,5 s | 110 s | 1.709 s | 94,5% |
| bola | 150 | 10,7 s | 33,8 s | 221,5 s | 85,2% |

Nenhuma lacuna de jogador dura menos de 2 s, e só 0,1% dos frames sem bola estão em lacunas de até
1 s. As lacunas são longas, não oclusões curtas, então **interpolar a trajetória da bola recupera
quase nada**. Esse resultado motivou a frente de tracking próprio da bola (`tracking_bola/`).

**Seção 6, duplicatas** (dois slots a menos de 0,5 m no mesmo frame):
- frames com um par do mesmo time: 3,47%;
- frames com mais de 22 objetos que têm um par do mesmo time: só 4,0%. As duplicatas a menos de
  0,5 m **não** explicam o excesso de objetos;
- pares mais frequentes: 40–45 (198 frames), 10–21 (157) e 15–19 (157), todos do mesmo time; e
  **24–46 (147, times diferentes)**.

**Seção 7, coerência espacial (descartada).** Mede a fração de frames em que um slot está mais
perto do centroide do outro time do que do próprio. A mediana foi 41,5%, porque os blocos dos dois
times se sobrepõem. O teste **não separa** slot certo de slot errado.

**Seção 8, espelhamento.** O centroide x médio da casa foi 0,467 no 1º tempo e 0,528 no 2º. Os
valores são próximos demais de 0,5 para concluir algo, e a pergunta foi resolvida no passo 4.

### Passo 2 — Ver o slot 24 no vídeo (`assistir_slot.py`)

O slot 24 chamou atenção na seção 4 (todos os saltos), na 6 (par com o 46) e na 1 (é o 24º slot
da casa). Para ver quem ele é:

```bash
python diagnostico/assistir_slot.py --slot 24 --inicio 30000
```

O script projeta cada slot e a bola no vídeo com a homografia e desenha:
- casa em vermelho, fora em azul, a bola em branco;
- o slot escolhido em amarelo, maior;
- no topo, o frame, o estado do slot (na tela / fora da tela / ausente) e a contagem por time.

Com `--slot`, ele começa no primeiro frame, a partir de `--inicio`, em que o slot está **dentro
da imagem**. Os argumentos e as teclas estão documentados no topo do arquivo.

**O que observamos:**
- **Presente no ATD não é o mesmo que visível no vídeo.** O slot 24 está presente em 178.478 de
  199.319 frames. No frame 30.000 a posição dele é (0,144; 0,495), que projeta em x = −1.762 px,
  fora da imagem de 1920 × 1080. A câmera tactical não mostra o campo inteiro, mas o ATD guarda
  posições de quem está fora dela.
- **O slot 24 é o goleiro do Barcelona.**
- **Em alguns momentos, o slot 46 (time "fora") passa a seguir o goleiro do Barcelona.** Nesses
  trechos o slot 24 fica parado onde o goleiro estava, como um "fantasma".

Os 147 frames com 24 e 46 colados (seção 6) são compatíveis com o momento dessa troca.

### Passo 3 — Trechos com contagem anormal (`listar_contagem.py`)

```bash
python diagnostico/listar_contagem.py casa --min 12
python diagnostico/listar_contagem.py fora --min 12
python diagnostico/listar_contagem.py casa --max 2
```

Lista os 20 trechos graduados mais longos em que um time tem slots demais (`--min`) ou de menos
(`--max`), com os slots presentes no início de cada trecho. O início vai direto em
`assistir_slot.py --slot N --inicio F --exato`.

Hipóteses para mais de 11 slots, a conferir no vídeo:
- duplicata acima de 0,5 m;
- alguém que não é jogador em campo contado no time (reserva, substituído);
- slot com o time errado depois de uma troca de identidade.


### Passo 4 — Slots de goleiro (`verificar_goleiro.py`)

```bash
python diagnostico/verificar_goleiro.py
```

Para cada slot com pelo menos 1.000 frames graduados, mede:
- **mais recuado:** % do tempo em que é o jogador mais perto do próprio gol;
- **na área:** % do tempo dentro da própria grande área (16,5 × 40,3 m);
- a distância mediana à própria linha de fundo.

O gol que cada time defende é decidido no frame do kickoff de cada tempo (11.185 e 113.742),
quando todos estão no próprio campo.

| slot | time | frames | mais recuado | na área | distância mediana |
|---|---|---|---|---|---|
| 24 | casa | 147.627 | 96,2% | 92,0% | 11,0 m |
| 46 | fora | 146.907 | 85,8% | 86,2% | 7,5 m |
| 42 (o seguinte) | fora | 50.045 | 7,0% | 9,2% | 43,0 m |

**Conclusões:**
- **Os slots 24 e 46 são os goleiros** do Barcelona e do Athletic. Nenhum outro slot passa de 7%
  como mais recuado.
- **Os times trocam de lado no 2º tempo.** Se o lado defendido estivesse errado em algum tempo, o
  slot 24 ficaria perto de 50% como mais recuado, e não em 96%.
- A diferença entre 96% e 86% é compatível com o que vimos no vídeo: nos trechos em que o 46
  segue o goleiro do Barcelona, ele está do lado errado do campo. Isso é hipótese, não foi medido.

---

## 3. Assets usados

| asset | uso |
|---|---|
| ATD tactical (`BAR-ATH_tactical_atd.txt`) | posições de slots e bola |
| Metadados do ATD (`BAR-ATH_tactical_atd.xml`) | time de cada slot, via `starter/baseline.team_of_each_track` |
| Homografia tactical | projetar o ATD (campo 0–1) em pixels do vídeo |
| Vídeo tactical | conferência visual com `assistir_slot.py` |
| `games.py` | janelas graduadas, dimensões do campo (105,3 × 68,0 m), caminhos dos dados |

**Testes descartados:** coerência espacial (seção 7), por não separar slots certos de errados, e o
espelhamento pelo centroide médio (seção 8), substituído pelo kickoff no passo 4.

**Não usados nesta parte:** o ATD e o vídeo do broadcast, que estão em outra linha do tempo e não
têm mapeamento para o vídeo tactical.

---

## 4. Implicações para o modelo de posse

1. **"Jogador mais próximo da bola" pode dar o time errado** perto do gol do Barcelona: nos
   trechos de troca, o goleiro do Barcelona aparece como slot 46, do time "fora", e o fantasma do
   slot 24 pode ser o mais próximo de uma bola que passa por ali.
2. **Posições fora da imagem não são observações garantidas.** Elas podem ser estimadas pelo
   tracker. Features sobre a forma do time ou o jogador mais próximo devem tratá-las com cautela.
3. **A ausência de bola não basta como sinal de bola parada** (`D`): a bola está presente em boa
   parte dos trechos sem jogo.
4. **A bola fora de [0,1]** pode ajudar a detectar `D`, desde que se descartem valores muito fora
   do campo.

---

## 5. Limitações

- **Os limiares são escolhas nossas, não vêm dos docs:** salto de 2 m, velocidade máxima de
  10 m/s, duplicata a 0,5 m. Podem ser ajustados com `--jump-m`, `--max-speed` e `--dup-m`.
- **Os testes indicam suspeitas, não confirmam erros.** A identidade de um slot só é confirmada
  olhando o vídeo.
- **Não sabemos como o tracker produz posições fora da imagem** (congelando, extrapolando ou com
  outra fonte). O teste comparando o movimento dentro e fora da tela não foi feito.
- **A troca 24/46 foi observada no vídeo, mas não medida.** Não sabemos em quantos frames ela
  ocorre.
- **A busca de frame do OpenCV** pode ser imprecisa em alguns codecs. O `assistir_slot.py` avisa
  quando a posição reportada difere da pedida.

---

## 6. Reprodução

```bash
pip install -r requirements.txt
pip install opencv-python              # só para assistir_slot.py
# descompactar o bundle de dados em data/ (tactical/)

python diagnostico/diagnostico.py      # gera data/_cache/atd_xy_v2.npz; rode primeiro
python diagnostico/verificar_goleiro.py
python diagnostico/listar_contagem.py casa --min 12
python diagnostico/assistir_slot.py --slot 24 --inicio 30000
```

- **Dependências:** `numpy` em todos os scripts; o `diagnostico.py` também usa o
  `starter/baseline.py`, e o `assistir_slot.py` o `starter/load_homography.py`, que usa `pandas`.
  Os scripts encontram esses módulos sozinhos, a partir da própria pasta.
- **O que não vai para o repositório:** o cache em `data/_cache/` e os `.mp4` gravados com
  `--salvar` ficam em `data/`, que está no `.gitignore`. São dados derivados do tracking e do vídeo,
  e as regras não permitem publicá-los.