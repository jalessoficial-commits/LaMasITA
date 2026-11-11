# Épocas, instância viterbi

## Resumo

| rede | épocas | sementes | DFL (validação com as mesmas épocas) | janela, detector rígido | janela, detector suave | teto (GT, não vale) | janela, bola morta própria | por frame: morta/viva | por frame: H/A (bola viva) |
|---|---|---|---|---|---|---|---|---|---|
| tcn | 5 | 3 | **0,616** (0,618 / 0,618 / 0,611) | **0,721** (0,729 / 0,706 / 0,727) | **0,722** (0,741 / 0,718 / 0,706) | **0,741** (0,765 / 0,741 / 0,718) | **0,651** (0,663 / 0,642 / 0,649) | **0,931** (0,919 / 0,943 / 0,932) | **0,952** (0,957 / 0,952 / 0,947) |
| lstm | 5 | 3 | **0,615** (0,615 / 0,618 / 0,613) | **0,716** (0,739 / 0,703 / 0,706) | **0,706** (0,718 / 0,682 / 0,718) | **0,718** (0,729 / 0,694 / 0,729) | **0,654** (0,653 / 0,638 / 0,670) | **0,928** (0,916 / 0,933 / 0,933) | **0,950** (0,951 / 0,947 / 0,953) |

Cada célula: **média das sementes** (valor de cada semente, na ordem). Janela: spellF1. A validação usa as mesmas épocas do modelo final. Na tabela de detalhe, janela: spellF1 / macroF1 (posses previstas; GT 17).

## Cada modelo

| modelo | rede | épocas | semente | treino | DFL | rígido | suave | teto | própria | morta/viva por frame | H/A por frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn_f5_s20 | tcn | 5 | 20 | já treinado | 0,618 | 0,729 / 0,940 (17) | 0,741 / 0,956 (17) | 0,765 / 0,971 (17) | 0,663 / 0,905 (21) | 0,919 | 0,957 |
| tcn_f5_s21 | tcn | 5 | 21 | já treinado | 0,618 | 0,706 / 0,937 (17) | 0,718 / 0,953 (17) | 0,741 / 0,967 (17) | 0,642 / 0,905 (21) | 0,943 | 0,952 |
| tcn_f5_s22 | tcn | 5 | 22 | 5 min (GPU) | 0,611 | 0,727 / 0,933 (16) | 0,706 / 0,949 (17) | 0,718 / 0,964 (17) | 0,649 / 0,907 (20) | 0,932 | 0,947 |
| lstm_f5_s20 | lstm | 5 | 20 | 5 min (GPU) | 0,615 | 0,739 / 0,935 (16) | 0,718 / 0,953 (17) | 0,729 / 0,967 (17) | 0,653 / 0,909 (21) | 0,916 | 0,951 |
| lstm_f5_s21 | lstm | 5 | 21 | 4 min (GPU) | 0,618 | 0,703 / 0,932 (16) | 0,682 / 0,949 (17) | 0,694 / 0,964 (17) | 0,638 / 0,907 (20) | 0,933 | 0,947 |
| lstm_f5_s22 | lstm | 5 | 22 | 5 min (GPU) | 0,613 | 0,706 / 0,938 (17) | 0,718 / 0,956 (17) | 0,729 / 0,968 (17) | 0,670 / 0,922 (20) | 0,933 | 0,953 |
