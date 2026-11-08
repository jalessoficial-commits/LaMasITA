# Validação em outros jogos (dados abertos da Metrica)

O RULES.md proíbe rótulos manuais sobre o jogo do desafio, inclusive para validar, e proíbe usar os 5 minutos
públicos para escolher entre versões. Dados rotulados de **outros jogos** são permitidos. Esta pasta usa os
jogos de amostra 1 e 2 da Metrica ([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)):
tracking a 25 fps e eventos sincronizados, feitos por pessoas. **Precisam ser citados no documento técnico**
(o SUBMISSION.md pede a origem, quem rotulou e quanto).

## Como rodar

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/metrica-sports/sample-data ../metrica-sample-data
cd ../metrica-sample-data && git sparse-checkout set data/Sample_Game_1 data/Sample_Game_2 && cd -
python validacao_metrica/testar_dono_reinicio.py       # alguns minutos; METRICA_DIR muda a pasta dos dados
```

`metrica.py` monta um gabarito H/A/D por frame a partir dos eventos (a regra está no docstring). São 88 e 90
paradas e cerca de 370 posses por jogo, com posse média de 9-10 s.

## Dono do reinício (`analise/dono_reinicio.py`)

H/A do baseline oficial (jogador mais perto da bola, voto de ±50 frames), com as paradas do gabarito por cima,
como o notebook 03 faz hoje. Depois, o dono estimado de cada parada vale nos 2 s seguintes ao reinício, antes
da suavização. spellF1, média de 5 sementes nos cenários com bola apagada:

| cenário | jogo | baseline + D | + dono estimado | + dono verdadeiro (teto) |
|---|---|---|---|---|
| bola da Metrica (some em ~7% do jogo vivo), D exato | 1 | 0,543 | **0,582** | 0,582 |
| | 2 | 0,522 | **0,551** | 0,551 |
| bola apagada em 35% do jogo vivo, D exato | 1 | 0,444 | **0,482** | 0,491 |
| | 2 | 0,427 | **0,448** | 0,456 |
| bola apagada, início e fim do D com erro de até ±2 s | 1 | 0,410 | **0,440** | 0,451 |
| | 2 | 0,394 | **0,410** | 0,413 |

- O ganho fica entre **+0,016 e +0,039** de spellF1 e vai na mesma direção nos dois jogos e nos três cenários.
- Só apagar a evidência de dentro das paradas, sem o dono, não muda nada (±0,01). O ganho vem do dono.
- O dono estimado acerta 100% das paradas com a bola da Metrica e 92-93% com a bola apagada.
- Manter o dono por 2 s fica perto do melhor nos dois jogos: 1 s perde parte do ganho e 4 s praticamente empata.

Acerto de cada sinal quando ele responde (bola apagada, D exato):

| sinal | acerto | responde em |
|---|---|---|
| cobrador (jogador mais perto da bola no reinício) | 98% | ~80% das paradas |
| último chute antes da bola sair (bola fora) | 72-80% | ~85% |
| posse antes do apito (faltas) | 64-72% | ~95% |
| Smart Tagging | não existe nestes jogos | — |

O "último jogador perto da bola" antes da saída acertou só ~50%, porque perto da linha quase sempre há um
adversário colado. Por isso o sinal usa o último salto de velocidade da bola (o chute).

## Como ligar no jogo do desafio

O dono entra na evidência por frame, antes da suavização, porque é a suavização que espalha o time errado da
parada para dentro da posse seguinte. Com o detector do Nuno e o baseline (não testado no jogo do desafio, só
nos jogos da Metrica):

```python
import paradas as P, arvore as T, dono_reinicio as DR
dead, tabela = P.detectar_paradas(S)                       # tabela: inicio, fim, tipo
# raw: 'H'/'A' do jogador mais perto da bola por frame, 'D' sem bola; smooth: o voto de starter/baseline.py
donos = DR.donos_do_jogo(tabela, rotulos=smooth(raw))      # usa o ATD e o Smart Tagging
sub = T.combine_with_baseline(smooth(DR.aplicar_donos(raw, donos)), dead, S.graded)
```

Com outro classificador de H/A, `aplicar_donos` vale do mesmo jeito sobre a evidência dele por frame.

## Limites

- O gabarito vem de eventos, não de um rótulo por frame como o do desafio. A regra de posse segue o
  CHALLENGE.md, mas não é idêntica.
- O H/A é o do baseline. Um classificador de H/A melhor já captura parte desse ganho, então o efeito sobre ele
  tende a ser menor.
- Os parâmetros do módulo foram escolhidos olhando estes dois jogos; um terceiro jogo (o Sample Game 3, em
  formato EPTS) seria o teste limpo.
- No jogo do desafio entra também o Smart Tagging, que estes jogos não têm. Antes de confiar nele, vale conferir
  no viewer se o campo Team de um set piece é mesmo o time que cobra.
