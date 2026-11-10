# Classificadores, instância viterbi

| classificador | treino | validação DFL (spellF1 / macroF1) | janela, bola morta própria | janela, detector rígido | janela, detector suave |
|---|---|---|---|---|---|
| hgb | já treinado | 0,596 / 0,909 | 0,667 / 0,913 (13) | 0,690 / 0,931 (12) | 0,690 / 0,940 (12) |
| xgboost | 7 min (GPU) | 0,596 / 0,909 | 0,667 / 0,914 (13) | 0,690 / 0,931 (12) | 0,690 / 0,942 (12) |
| lightgbm | 8 min (CPU) | 0,591 / 0,909 | 0,645 / 0,906 (14) | 0,690 / 0,930 (12) | 0,690 / 0,940 (12) |
| gru | falhou | 0,613 / 0,916 | 0,613 / 0,913 (15) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) |
| tcn | 5 min (GPU) | 0,609 / 0,913 | 0,678 / 0,907 (19) | 0,727 / 0,933 (16) | 0,729 / 0,948 (17) |
| catboost | 3 min (GPU) | 0,572 / 0,904 | 0,625 / 0,914 (15) | 0,690 / 0,933 (12) | 0,667 / 0,943 (13) |

Janela pública: spellF1 / macroF1 (posses previstas; GT 17). Quem decide é a validação na DFL (7 jogos inteiros); a janela tem 17 posses e serve só para conferir.
