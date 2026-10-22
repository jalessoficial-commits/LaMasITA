# Viewer — vídeo + tracking + posse

Site local para assistir ao **vídeo tático** junto com o tracking (oficial e o nosso), um **mapa tático 2D** e o
**gabarito de posse**. Ele serve para inspecionar o modelo frame a frame e ver onde a predição erra.

## Instalação e uso

Pré-requisitos: Python 3.9+ e Chrome ou Edge, que tocam o H.264 do vídeo.

```bash
pip install -r viewer/requirements.txt
python viewer/preprocess.py       # uma vez, ~40 s: gera viewer/data/ a partir de data/
python viewer/serve.py            # abre http://localhost:8000/viewer/
```

Os dados da partida precisam estar em `data/`, no layout descrito em [DATA.md](../DATA.md):

```
data/tactical/
├─ *.mp4                                   o primeiro .mp4 da pasta, o nome não importa
├─ BAR-ATH_tactical_atd.xml / _atd.txt
├─ BAR-ATH_tactical_homography.csv
└─ BAR-ATH_tactical_smart_tagging.xml
```

Para usar outra pasta, use `MTAH_DATA_DIR` (a mesma variável do `games.py`) ou `--data-dir` nos dois scripts.
Para usar outra porta: `python viewer/serve.py --port 8001`.

> `viewer/data/` é gerado a partir dos dados restritos e está no `.gitignore`. Não faça commit dessa pasta.

## Funcionalidades

| | |
|---|---|
| **Caixa de posse** | Fica no topo, em destaque. Mostra o gabarito público (frames 11185–18684) com o escudo do time. Barcelona: grená/azul com borda dourada. Athletic: listras vermelho/branco. Bola morta: cinza. Sem gabarito: hachurado. |
| **Nossa predição** | CSV `frame,label`, no mesmo formato da submissão. Mostra ✓/✗ por frame e a acurácia na janela do gabarito. Os baselines do repositório estão no menu. |
| **Vídeo + overlay** | O tracking é projetado sobre o vídeo pela homografia de cada frame. A opção "linhas do campo" serve para conferir a homografia. |
| **Mapa 2D** | Campo de 105,3 × 68 m, com opção de inverter X e Y. |
| **Camadas** | Cada fonte de tracking (o ATD oficial e cada CSV extra) tem cor por time e para a bola, forma, tamanho, opacidade, IDs e rastro, e pode ser ligada no vídeo e/ou no 2D. Isso permite sobrepor as fontes. |
| **Timeline** | Linhas: gabarito · predição · erros (vermelho) · smart tagging. Clique ou arraste para navegar. O zoom pode mostrar a janela do gabarito, o 1º tempo, o 2º tempo ou o vídeo inteiro. |
| **Navegação** | Play/pause, ±1 frame, ±1 s, ±10 s, velocidade de 0,25× a 4×, ir para frame e ir para tempo (`mm:ss`). |

### Atalhos

| tecla | ação |
|---|---|
| `Espaço` | play / pause |
| `←` / `→` | ±1 frame |
| `Shift` + setas | ±1 s |
| `Ctrl` + setas | ±10 s |
| `[` / `]` | velocidade |
| `G` | início do gabarito |

## Formato do tracking extra

Carregue pelo botão **➕ Adicionar tracking (CSV)**. Você pode adicionar vários arquivos, e cada um vira uma camada.

```csv
frame,track_id,team,x,y
11185,7,H,0.4123,0.5531
11185,ball,B,0.5012,0.4990
```

| coluna | significado |
|---|---|
| `frame` | frame do vídeo tático, começando em 0 (= linha da submissão = `frame_id` do ATD − 1) |
| `track_id` | qualquer identificador; é usado nos rótulos e no rastro |
| `team` | `H` (Barcelona), `A` (Athletic) ou `B` (bola) |
| `x`, `y` | normalizados de 0 a 1, na mesma convenção do ATD (y = 0 é a lateral do fundo da imagem) |

## Como funciona

```
viewer/
├─ preprocess.py     ATD (195 MB de texto) → atd_tactical.bin (uint16, ~37 MB)
│                    homografia → homography_tactical.bin (float32, ~7 MB), smart tagging → JSON
├─ serve.py          http.server com suporte a Range (necessário para fazer seek no vídeo de 5 GB);
│                    expõe o vídeo em /video/tactical.mp4 e serve a raiz do repositório (para ler evaluation/)
├─ index.html        layout + estilos
├─ js/app.js         estado, relógio vídeo↔frame (requestVideoFrameCallback), UI, timeline
├─ js/loaders.js     binários, CSVs de posse e de tracking
├─ js/overlay.js     projeção pela homografia sobre o vídeo
├─ js/pitch.js       mapa 2D e desenho dos marcadores
└─ assets/           escudos (SVG da Wikimedia Commons)
```

O índice 0 de tudo é o frame 0 do vídeo. Ao fazer seek, o vídeo vai para o meio do frame (`(f + 0.5) / 25`), para que o arredondamento nunca caia no frame vizinho.
