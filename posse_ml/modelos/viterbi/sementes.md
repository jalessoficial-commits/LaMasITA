# Sementes, instância viterbi

| modelo | validação DFL (spellF1 / macroF1) | janela, bola morta própria | janela, detector rígido | janela, detector suave | teto (bola morta do GT, não vale) |
|---|---|---|---|---|---|
| tcn | 0,609 / 0,913 | 0,678 / 0,907 (19) | 0,727 / 0,933 (16) | 0,729 / 0,948 (17) | 0,729 / 0,965 (17) |
| tcn_s1 | 0,611 / 0,913 | 0,635 / 0,909 (17) | 0,671 / 0,935 (14) | 0,675 / 0,952 (15) | 0,675 / 0,967 (15) |
| tcn_s2 | 0,611 / 0,914 | 0,630 / 0,902 (16) | 0,667 / 0,933 (13) | 0,693 / 0,947 (13) | 0,693 / 0,964 (13) |
| tcn_sementes | 0,616 / 0,915 | 0,635 / 0,911 (17) | 0,671 / 0,936 (14) | 0,675 / 0,951 (15) | 0,675 / 0,968 (15) |
| gru | 0,613 / 0,916 | 0,613 / 0,913 (15) | 0,676 / 0,930 (12) | 0,653 / 0,945 (13) | 0,667 / 0,961 (13) |
| gru_s1 | 0,616 / 0,916 | 0,629 / 0,905 (18) | 0,679 / 0,924 (16) | 0,647 / 0,938 (17) | 0,671 / 0,955 (17) |
| gru_s2 | 0,612 / 0,915 | 0,606 / 0,912 (16) | 0,640 / 0,932 (13) | 0,680 / 0,945 (13) | 0,680 / 0,963 (13) |
| gru_sementes | 0,614 / 0,917 | 0,625 / 0,913 (15) | 0,676 / 0,930 (12) | 0,667 / 0,944 (13) | 0,667 / 0,961 (13) |

- tcn rigido: spellF1 na janela de 0,667 a 0,727 (média 0,688, 3 sementes)
- tcn suave: spellF1 na janela de 0,675 a 0,729 (média 0,699, 3 sementes)
- tcn teto: spellF1 na janela de 0,675 a 0,729 (média 0,699, 3 sementes)
- tcn DFL: spellF1 de 0,609 a 0,611 (média 0,610)
- gru rigido: spellF1 na janela de 0,640 a 0,679 (média 0,665, 3 sementes)
- gru suave: spellF1 na janela de 0,647 a 0,680 (média 0,660, 3 sementes)
- gru teto: spellF1 na janela de 0,667 a 0,680 (média 0,672, 3 sementes)
- gru DFL: spellF1 de 0,612 a 0,616 (média 0,613)

Janela pública: spellF1 / macroF1 (posses previstas; GT 17). Quem decide é a validação na DFL (7 jogos inteiros); a janela tem 17 posses e serve só para conferir.
