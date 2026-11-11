# Dados de treino (3 épocas), instância viterbi

| rede | treino | épocas | sementes | validação (jogos de treino) | janela, rígido | macroF1, rígido | posses no rígido (GT 17) | janela, suave | teto (GT, não vale) | janela, própria | por frame: morta/viva | por frame: H/A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | dfl (referência) | 5 | 3 | **0,618** | **0,728** (0,729 / 0,727 / 0,727) | **0,937** | **16,3** | **0,733** (0,753 / 0,729 / 0,718) | **0,749** | **0,676** | **0,933** | **0,953** |
| tcn | dfl+metrica | 3 | 3 | **0,641** | **0,730** (0,727 / 0,723 / 0,739) | **0,935** | **15,3** | **0,724** (0,729 / 0,725 / 0,718) | **0,732** | **0,682** | **0,918** | **0,953** |
| lstm | dfl (referência) | 5 | 3 | **0,615** | **0,717** (0,741 / 0,706 / 0,703) | **0,937** | **16,7** | **0,718** (0,753 / 0,718 / 0,682) | **0,737** | **0,673** | **0,934** | **0,953** |
| lstm | dfl+metrica | 3 | 3 | **0,644** | **0,699** (0,739 / 0,713 / 0,645) | **0,936** | **15,0** | **0,705** (0,741 / 0,725 / 0,650) | **0,718** | **0,644** | **0,920** | **0,952** |

Cada célula: **média das sementes**; no rígido e no suave, também o valor de cada semente. Janela: spellF1. A coluna de validação usa os jogos de treino de cada linha e não se compara entre linhas. Uma semente oscila ~0,02 a 0,04 na janela: menos de 0,03 de diferença é empate.
