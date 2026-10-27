# Resumo: artigos e aplicações no desafio

Oct 26, 2026 · @Pedro Araujo

Este resumo mostra, para cada artigo, o método usado e como ele pode ser aplicado no desafio de posse de bola do More Than A Hack 2027. A análise completa, com resultados e limitações, está no documento “Análise dos artigos sobre posse de bola”.

## Os cinco artigos

Cada artigo resolve uma parte do problema. Juntos, eles cobrem a posse, a bola morta e a bola ausente, mas nenhum transforma as decisões por frame em posses inteiras, que é o que o spellF1 avalia.

| Artigo | Método | Aplicação no desafio |
| --- | --- | --- |
| [Borghesi et al. (2023)](https://iris.polito.it/retrieve/8cc27103-7ee0-4243-8a04-0b256f77ad39/eswa_tcn_authors_accepted_version.pdf) | Uma rede neural temporal (TCN) lê cerca de 4 s de posições da bola e dos jogadores e diz, a cada frame, se a bola está morta, com a casa ou com o visitante. Acertou 86,2% na posse e 89,2% em bola viva ou morta. | Ser o classificador central, que dá a probabilidade de H, A e D em cada frame. É também a melhor base para detectar bola morta pelo movimento dos jogadores. |
| [Vidal-Codina et al. (2022)](https://arxiv.org/abs/2202.00804) | Regras fixas: a bola pertence ao jogador quando entra num círculo em volta dele, e um ganho só vale se a bola mudar de direção ou de velocidade. A posição dos jogadores durante a parada indica o tipo de reinício. | Servir de baseline de regras e decidir quando uma troca de posse é verdadeira. Também indica qual time reinicia o jogo, inclusive nos laterais que o Smart Tagging deixa sem time. |
| [Kim et al. (2023), ballradar](https://arxiv.org/pdf/2306.08206) | Um Set Transformer e redes Bi-LSTM estimam quem tem a bola e onde ela está usando só os jogadores. Acertou o time com a bola em 85,85% dos casos. | Dar evidência de posse nos 41% de frames em que a bola não aparece no tracking. |
| [Link & Hoernig (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5503225) | Detecta chutes por picos de aceleração da bola, calcula o alcance de cada jogador e usa uma rede bayesiana para separar domínio de toque. | Confirmar ou rejeitar trocas de posse, porque o desafio só troca de time quando o adversário domina a bola. |
| [ELASTIC (2025)](https://arxiv.org/html/2508.09238v1) | Acha o frame exato de eventos anotados usando distância jogador-bola e aceleração da bola. | Ajustar as 59 bolas paradas do Smart Tagging ao frame exato do chute de reinício, que marca a passagem de D para H ou A. |

## Outros modelos com mais potencial

Dos 14 candidatos encontrados fora dos cinco artigos, estes quatro são os que mais podem melhorar o resultado.

| Modelo | Método | Aplicação no desafio |
| --- | --- | --- |
| [PathCRF (KDD 2026)](https://arxiv.org/pdf/2602.12080) | Reconstrói o caminho da bola só com os jogadores: a cada instante, decide se um jogador tem a bola ou se ela vai de um jogador para outro. Um modelo de sequência (CRF) proíbe passagens impossíveis. Acertou o time com a posse em 92,4% dos frames de um jogo de teste. | Cobrir os frames sem bola e, ao mesmo tempo, manter as posses contínuas. Exige ligar antes as trajetórias de cada jogador, porque os slots do ATD trocam de pessoa. |
| [MS-TCN++](https://ar5iv.labs.arxiv.org/html/2006.09220) e [ASRF](https://ar5iv.labs.arxiv.org/html/2007.06866) | Vêm da segmentação de ações em vídeo. O MS-TCN++ penaliza mudanças bruscas entre frames vizinhos, e o ASRF detecta as fronteiras entre trechos e vota o rótulo dentro de cada um. | Reduzir posses falsas e curtas. Esses métodos foram feitos para uma métrica por trechos parecida com o spellF1. |
| Dados abertos da Sportec (7 jogos) | Os arquivos de posição da liga alemã trazem, segundo a documentação do floodlight, códigos de posse e de bola em jogo por frame. | Treinar e validar fora dos 5 minutos públicos, com rótulos da mesma natureza do ground truth, se os códigos estiverem presentes. |
| [WASB](https://arxiv.org/pdf/2311.05237) | Detector de bola em vídeo que olha 3 frames seguidos. Teve F1 de 88,3% em vídeo de futebol de câmeras fixas. | Recuperar a bola no vídeo tático nos frames em que o tracking a perdeu, usando a homografia para converter a posição para o campo. |

## Como os métodos se combinam

Os métodos funcionam como etapas de um mesmo sistema. Esta combinação é proposta minha e ainda não foi testada nos dados do desafio.

1. **Preparar os dados.** Converter as posições para metros, suavizar a trajetória da bola e preencher só as lacunas curtas.
2. **Gerar evidências por frame.** O classificador de Borghesi dá a probabilidade de H, A e D. O ballradar ou o PathCRF cobrem os frames sem bola. O detector de domínio de Link & Hoernig e as regras de Vidal-Codina indicam quando o adversário dominou a bola.
3. **Marcar as paradas.** O movimento dos jogadores indica que há uma parada. A saída da bola do campo marca o início dela, e o chute de reinício, ajustado como no ELASTIC, marca o fim.
4. **Decodificar em posses.** O algoritmo de Viterbi escolhe a sequência de estados mais provável, com um custo para cada troca. O custo de trocar entre H e A cai quando há evidência de domínio do adversário, o que segue a regra do desafio. A penalidade do MS-TCN++ ou as fronteiras do ASRF podem reforçar essa etapa.
