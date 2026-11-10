"""Posse frame a frame com classificadores treinados na DFL e a bola do nosso tracker.

Ordem do pipeline: partida (ATD + nossa bola) -> bola (limpeza) -> atributos -> modelos
(probabilidades por frame) -> transmissao (plano da TV) -> viterbi (posses inteiras).

    python -m posse_ml        escreve a submissão (veja posse_ml/README.md)
"""
