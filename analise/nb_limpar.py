"""Filtro do Git que tira as saídas dos notebooks no commit; a cópia local continua com elas.

Os gráficos interativos (plotly) embutem os dados frame a frame, e o RULES.md proíbe publicar dado
derivado que reconstrua os dados da partida. Ativar uma vez por clone:

    git config filter.nb_limpar.clean "python analise/nb_limpar.py"
    git config filter.nb_limpar.smudge cat

O .gitattributes já aplica o filtro a analise/*.ipynb. Também roda direto: python analise/nb_limpar.py < a.ipynb
"""

import json
import sys


def limpar(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    return nb


def main():
    text = sys.stdin.buffer.read().decode("utf-8")
    nb = limpar(json.loads(text))
    out = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    sys.stdout.buffer.write(out.encode("utf-8"))


if __name__ == "__main__":
    main()
