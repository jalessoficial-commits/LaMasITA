# Variantes das redes, instância viterbi

| modelo | épocas (validação / final) | treino | validação DFL (spellF1 / macroF1) | janela, bola morta própria | janela, detector rígido | janela, detector suave | teto (bola morta do GT, não vale) |
|---|---|---|---|---|---|---|---|
| tcn (referência) | 5 / 10 | já treinado | 0,609 / 0,913 | 0,678 / 0,907 (19) | 0,727 / 0,933 (16) | 0,729 / 0,948 (17) | 0,729 / 0,965 (17) |
| gru (referência) | 5 / 10 | já treinado | 0,613 / 0,916 | 0,613 / 0,913 (15) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) | 0,667 / 0,961 (13) |
| lstm | 5 / 10 | já treinado | 0,619 / 0,915 | 0,674 / 0,897 (18) | 0,727 / 0,931 (16) | 0,715 / 0,944 (16) | 0,715 / 0,958 (16) |
| tcn_e10 | 10 / 10 | já treinado | 0,633 / 0,918 | 0,635 / 0,910 (17) | 0,671 / 0,936 (14) | 0,675 / 0,951 (15) | 0,675 / 0,968 (15) |
| tcn_e20 | 20 / 20 | já treinado | 0,658 / 0,926 | 0,588 / 0,905 (17) | 0,619 / 0,924 (14) | 0,613 / 0,939 (15) | 0,625 / 0,955 (15) |
| tcn_e30 | 30 / 30 | já treinado | 0,661 / 0,926 | 0,576 / 0,907 (17) | 0,594 / 0,919 (14) | 0,613 / 0,934 (15) | 0,613 / 0,951 (15) |
| transformer | 5 / 10 | 10 min (GPU) | 0,595 / 0,910 | 0,663 / 0,934 (18) | 0,700 / 0,944 (15) | 0,713 / 0,955 (15) | 0,725 / 0,974 (15) |

Janela pública: spellF1 / macroF1 (posses previstas; GT 17). Quem decide é a validação na DFL (7 jogos inteiros); a janela tem 17 posses e serve só para conferir.
