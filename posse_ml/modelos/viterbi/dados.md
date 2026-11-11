# Dados de treino (5 épocas), instância viterbi

| rede | treino | sementes | validação (jogos de treino) | janela, rígido | macroF1, rígido | posses no rígido (GT 17) | janela, suave | teto (GT, não vale) | janela, própria | por frame: morta/viva | por frame: H/A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | dfl | 3 | **0,618** | **0,728** (0,729 / 0,727 / 0,727) | **0,937** | **16,3** | **0,733** (0,753 / 0,729 / 0,718) | **0,749** | **0,676** | **0,933** | **0,953** |
| tcn | metrica | 3 | **0,724** | **0,678** (0,652 / 0,731 / 0,652) | **0,929** | **10,7** | **0,674** (0,652 / 0,717 / 0,652) | **0,674** | **0,628** | **0,931** | **0,941** |
| tcn | dfl+metrica | 3 | **0,656** | **0,723** (0,703 / 0,727 / 0,739) | **0,936** | **14,7** | **0,708** (0,680 / 0,718 / 0,727) | **0,713** | **0,648** | **0,917** | **0,952** |
| lstm | dfl | 3 | **0,615** | **0,717** (0,741 / 0,706 / 0,703) | **0,937** | **16,7** | **0,718** (0,753 / 0,718 / 0,682) | **0,737** | **0,673** | **0,934** | **0,953** |
| lstm | metrica | 3 | **0,721** | **0,648** (0,640 / 0,652 / 0,652) | **0,932** | **11,0** | **0,657** (0,667 / 0,652 / 0,652) | **0,651** | **0,613** | **0,926** | **0,942** |
| lstm | dfl+metrica | 3 | **0,657** | **0,686** (0,715 / 0,710 / 0,634) | **0,932** | **14,0** | **0,681** (0,703 / 0,688 / 0,653) | **0,695** | **0,619** | **0,905** | **0,950** |

Cada célula: **média das sementes**; no rígido e no suave, também o valor de cada semente. Janela: spellF1. A coluna de validação usa os jogos de treino de cada linha e não se compara entre linhas. Uma semente oscila ~0,02 a 0,04 na janela: menos de 0,03 de diferença é empate.
