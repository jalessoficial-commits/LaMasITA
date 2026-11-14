"""No modo rígido, a calibração da janela pública não muda a decodificação (e por isso o prever.py
não a usa). Sem dados: probabilidades sorteadas.

    python -m unittest posse_ml.testes.test_rigido
"""

import unittest

import numpy as np

from posse_ml import viterbi


def _caso(rng):
    n = int(rng.integers(300, 3000))
    probs = [np.clip(rng.random(n), 0.01, 0.99) for _ in range(3)]
    m = n // 2
    tempos = [(1, 5, m - 5), (2, m + 5, n - 5)]
    kickoffs = {1: int(rng.integers(5, 100)), 2: int(rng.integers(m + 5, m + 100))}
    forca = np.zeros(n, np.int8)
    for _ in range(int(rng.integers(1, 15))):
        s = int(rng.integers(0, n))
        forca[s:s + int(rng.integers(10, 300))] = 1
    custos = dict(viterbi.CUSTOS, troca=float(rng.uniform(1, 10)))
    return probs, tempos, kickoffs, forca, custos, rng.normal(0, 3, n)


class TestRigido(unittest.TestCase):
    def test_extra_nao_muda_o_rigido(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            probs, tempos, ko, forca, c, extra = _caso(rng)
            sem = viterbi.decodificar(viterbi.emissoes(*probs), tempos, c, ko, forca)
            com = viterbi.decodificar(viterbi.emissoes(*probs, extra_morta=extra), tempos, c, ko, forca)
            np.testing.assert_array_equal(sem, com)

    def test_extra_muda_sem_forcar(self):
        """Controle: sem a bola morta fixada, o extra muda o resultado (o teste acima não é vazio)."""
        rng = np.random.default_rng(0)
        probs, tempos, ko, _, c, extra = _caso(rng)
        sem = viterbi.decodificar(viterbi.emissoes(*probs), tempos, c, ko)
        com = viterbi.decodificar(viterbi.emissoes(*probs, extra_morta=extra), tempos, c, ko)
        self.assertTrue(np.any(sem != com))


if __name__ == "__main__":
    unittest.main()
