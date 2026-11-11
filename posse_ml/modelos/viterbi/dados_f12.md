# Dados de treino (12 épocas), instância viterbi

| rede | treino | épocas | sementes | validação (jogos de treino) | janela, rígido | macroF1, rígido | posses no rígido (GT 17) | janela, suave | teto (GT, não vale) | janela, própria | por frame: morta/viva | por frame: H/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | dfl (referência) | 5 | 3 | **0,618** | **0,728** (0,729 / 0,727 / 0,727) | **0,937** | **16,3** | **0,733** (0,753 / 0,729 / 0,718) | **0,749** | **0,676** | **0,933** | **0,953** |
| tcn | metrica | 12 | 3 | **0,745** | **0,693** (0,731 / 0,681 / 0,667) | **0,934** | **10,7** | **0,688** (0,717 / 0,681 / 0,667) | **0,688** | **0,660** | **0,929** | **0,945** |
| lstm | dfl (referência) | 5 | 3 | **0,615** | **0,717** (0,741 / 0,706 / 0,703) | **0,937** | **16,7** | **0,718** (0,753 / 0,718 / 0,682) | **0,737** | **0,673** | **0,934** | **0,953** |
| lstm | metrica | 12 | 3 | **0,745** | **0,669** (0,652 / 0,703 / 0,652) | **0,928** | **10,7** | **0,669** (0,652 / 0,703 / 0,652) | **0,669** | **0,607** | **0,920** | **0,936** |

Cada célula: **média das sementes**; no rígido e no suave, também o valor de cada semente. Janela: spellF1. A coluna de validação usa os jogos de treino de cada linha e não se compara entre linhas. Uma semente oscila ~0,02 a 0,04 na janela: menos de 0,03 de diferença é empate.
