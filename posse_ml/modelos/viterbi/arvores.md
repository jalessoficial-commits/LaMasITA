# Árvores (xgboost), instância viterbi

| modelo | árvores (validação / final) | folhas | treino | validação DFL (spellF1 / macroF1) | janela, bola morta própria | janela, detector rígido | janela, detector suave | teto (bola morta do GT, não vale) | por frame: morta/viva | por frame: H/A (bola viva) |
|---|---|---|---|---|---|---|---|---|---|---|
| xgboost (referência) | 150 / 400 (padrão) | 63 | já treinado | 0,596 / 0,909 | 0,667 / 0,914 (13) | 0,690 / 0,931 (12) | 0,690 / 0,942 (12) | 0,690 / 0,959 (12) | 0,915 | 0,943 |
| xgboost_a50_f15 | 50 / 50 | 15 | 14 min (GPU) | 0,538 / 0,889 | 0,625 / 0,920 (15) | 0,662 / 0,931 (12) | 0,653 / 0,944 (13) | 0,645 / 0,959 (14) | 0,910 | 0,940 |
| xgboost_a150_f15 | 150 / 150 | 15 | 8 min (GPU) | 0,574 / 0,903 | 0,640 / 0,916 (13) | 0,662 / 0,930 (12) | 0,662 / 0,940 (12) | 0,680 / 0,963 (13) | 0,913 | 0,942 |
| xgboost_a400_f15 | 400 / 400 | 15 | 16 min (GPU) | 0,596 / 0,910 | 0,632 / 0,912 (14) | 0,676 / 0,931 (12) | 0,676 / 0,941 (12) | 0,676 / 0,958 (12) | 0,917 | 0,939 |
| xgboost_a50_f63 | 50 / 50 | 63 | 11 min (GPU) | 0,562 / 0,899 | 0,627 / 0,923 (13) | 0,648 / 0,932 (12) | 0,648 / 0,942 (12) | 0,648 / 0,959 (12) | 0,902 | 0,946 |
| xgboost_a150_f63 | 150 / 150 | 63 | 15 min (GPU) | 0,596 / 0,909 | 0,640 / 0,912 (13) | 0,662 / 0,930 (12) | 0,662 / 0,940 (12) | 0,662 / 0,958 (12) | 0,914 | 0,943 |
| xgboost_a400_f63 | 400 / 400 | 63 | 80 min (GPU) | 0,603 / 0,913 | 0,667 / 0,914 (13) | 0,690 / 0,931 (12) | 0,690 / 0,942 (12) | 0,690 / 0,959 (12) | 0,915 | 0,943 |

Janela pública: spellF1 / macroF1 (posses previstas; GT 17). Quem decide é a validação na DFL (7 jogos inteiros); a janela tem 17 posses e serve só para conferir.
