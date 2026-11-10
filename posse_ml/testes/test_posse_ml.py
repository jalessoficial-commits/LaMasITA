"""Testes que não precisam dos dados do desafio nem da DFL.

    python -m unittest discover -s posse_ml/testes -t .
"""

import os
import tempfile
import unittest

import numpy as np

from posse_ml import (atributos, config, bola, degradacao, dfl, janelas as J, modelos, partida, perfil,
                      transmissao, viterbi)
from posse_ml.config import A, D, FPS, H, MANDANTE, VISITANTE, X
from posse_ml.testes import sintetico


def viterbi_lento(pontos, custo):
    n, k = pontos.shape
    melhor = pontos[0].copy()
    volta = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        novo = np.empty(k)
        for j in range(k):
            de, val = j, melhor[j]
            for i in range(k):
                if melhor[i] - custo[i, j] > val:
                    de, val = i, melhor[i] - custo[i, j]
            novo[j] = val + pontos[t, j]
            volta[t, j] = de
        melhor = novo
    out = np.empty(n, dtype=int)
    out[-1] = int(np.argmax(melhor))
    for t in range(n - 1, 0, -1):
        out[t - 1] = volta[t, out[t]]
    return out


class Janelas(unittest.TestCase):
    def test_medias_contra_forca_bruta(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        x[rng.random(300) < 0.3] = np.nan
        for i in (0, 5, 150, 299):
            for w in (1, 4, 25):
                lo, hi = max(0, i - w), min(300, i + w + 1)
                for got, fatia in ((J.media_centrada(x, w)[i], x[lo:hi]), (J.media_antes(x, w)[i], x[lo:i]),
                                   (J.media_depois(x, w)[i], x[i + 1:hi])):
                    fatia = fatia[np.isfinite(fatia)]
                    if len(fatia):
                        self.assertAlmostEqual(got, fatia.mean())
                    else:
                        self.assertTrue(np.isnan(got))

    def test_desde_e_ate(self):
        m = np.array([0, 1, 0, 0, 1, 0, 0], dtype=bool)
        np.testing.assert_array_equal(J.desde(m, 99), [99, 0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(J.ate(m, 99), [1, 0, 2, 1, 0, 99, 99])

    def test_trechos(self):
        np.testing.assert_array_equal(J.trechos([0, 1, 1, 0, 1]), [[1, 3], [4, 5]])


class Bola(unittest.TestCase):
    def test_pulo_isolado_sai_e_reta_fica(self):
        b = np.column_stack([np.linspace(0, 30, 50), np.full(50, 10.0)]).astype(np.float32)
        b[20] += [0, 12]
        b[21] += [0, 12]
        limpa, fora = bola.remover_pulos(b)
        self.assertTrue(fora[20] and fora[21])
        self.assertEqual(int(fora.sum()), 2)
        self.assertTrue(np.isfinite(limpa[[19, 22], 0]).all())

    def test_ponte_so_nas_curtas(self):
        b = np.column_stack([np.arange(40.0), np.zeros(40)]).astype(np.float32)
        b[5:10] = np.nan          # 5 frames: preenche
        b[20:30] = np.nan         # 10 frames: não
        out = bola.ponte(b)
        np.testing.assert_allclose(out[5:10, 0], np.arange(5, 10))
        self.assertTrue(np.isnan(out[20:30, 0]).all())


class Viterbi(unittest.TestCase):
    def test_igual_a_versao_lenta(self):
        rng = np.random.default_rng(1)
        for _ in range(4):
            pts = np.log(rng.dirichlet(np.ones(4) * 0.5, size=400))
            c = viterbi.matriz_custos(*rng.uniform(1, 10, 3))
            np.testing.assert_array_equal(viterbi.caminho(pts, c), viterbi_lento(pts, c))

    def test_grade_igual_a_um_por_um(self):
        rng = np.random.default_rng(2)
        pts = np.log(rng.dirichlet(np.ones(4), size=500))
        grade = [{"troca": a, "parada": 4.0, "reinicio": 1.0} for a in (2.0, 9.0)]
        juntos = viterbi.decodificar(pts, [(1, 0, 499)], grade)
        for g, c in enumerate(grade):
            np.testing.assert_array_equal(juntos[g], viterbi.decodificar(pts, [(1, 0, 499)], c))

    def test_morta_so_sai_para_o_proprio_time(self):
        c = viterbi.matriz_custos(5, 5, 1)
        self.assertGreaterEqual(c[viterbi.MORTA_M, viterbi.VIVA_V], viterbi.PROIBIDO)
        self.assertGreaterEqual(c[viterbi.MORTA_V, viterbi.VIVA_M], viterbi.PROIBIDO)

    def test_kickoff(self):
        pts = np.log(np.full((300, 4), 0.25))
        pts[:, viterbi.VIVA_M] += 1.0      # tudo pede bola viva
        est = viterbi.decodificar(pts, [(1, 0, 299)], kickoffs={1: 100})
        rot = viterbi.rotulos(est)
        self.assertTrue((rot[:100] == D).all())
        self.assertTrue((rot[100:100 + FPS] != D).all())

    def test_forcar_paradas(self):
        pts = np.log(np.full((300, 4), 0.25))
        pts[:, viterbi.VIVA_M] += 1.0                      # o modelo quer bola viva o tempo todo
        forca = np.full(300, -1, dtype=np.int8)
        forca[150:200] = 1                                  # a fonte de fora diz parada aqui
        forca[0:50] = 1                                     # e aqui, mas antes do kickoff + 1 s...
        rot = viterbi.rotulos(viterbi.decodificar(pts, [(1, 0, 299)], kickoffs={1: 20}, forcar=forca))
        self.assertTrue((rot[150:200] == D).all())
        self.assertTrue((rot[:20] == D).all())
        self.assertTrue((rot[20:20 + FPS] != D).all())      # ...o kickoff manda
        self.assertTrue((rot[200:] != D).all())

    def test_adiantar_paradas(self):
        rot = np.array([H] * 100 + [D] * 100 + [A] * 100, dtype=np.int8)
        out = viterbi.adiantar_paradas(rot, [(1, 0, 299)], 25, 12)
        self.assertTrue((out[75:188] == D).all())
        self.assertTrue((out[188:200] == A).all() and (out[:75] == H).all())
        curta = np.array([H] * 10 + [D] * 100 + [A] * 10, dtype=np.int8)
        out = viterbi.adiantar_paradas(curta, [(1, 0, 119)], 25, 0)
        self.assertEqual(int((out == H).sum()), 5)      # nunca come mais da metade da posse


class Leitura(unittest.TestCase):
    def test_atd_bola_e_goleiros(self):
        p = sintetico.jogo(60, semente=3)
        with tempfile.TemporaryDirectory() as d:
            xml, txt, csv = (os.path.join(d, n) for n in ("m.xml", "r.txt", "b.csv"))
            sintetico.escrever_atd(p, xml, txt)
            sintetico.escrever_bola_csv(p.bola, p.campo, csv)
            time = partida.times_dos_slots(xml)
            np.testing.assert_array_equal(time, p.lado[0])
            jog, b = partida.ler_atd(txt, p.n, len(time), usar_cache=False)
            np.testing.assert_allclose(jog * np.array(p.campo), p.jogadores, atol=1e-3, equal_nan=True)
            np.testing.assert_allclose(partida.ler_bola_csv(csv, p.n) * np.array(p.campo), p.bola,
                                       atol=1e-3, equal_nan=True)
        # o slot do goleiro do visitante (11) passa a seguir o goleiro do mandante por 400 frames
        q = p.trocar(jogadores=p.jogadores.copy())
        a = p.tempos[0][1] + 2000
        q.jogadores[a:a + 400, 11] = q.jogadores[a:a + 400, 0]
        q.jogadores[a:a + 400, 0] = np.nan
        corr, mudou = partida.corrigir_goleiros(q)
        self.assertTrue((corr.lado[a:a + 400, 11] == MANDANTE).all())
        self.assertTrue((corr.lado[a + 400:a + 800, 11] == VISITANTE).all())
        self.assertEqual(mudou[11], 400)

    def test_dfl(self):
        p = sintetico.jogo(40, semente=4)
        with tempfile.TemporaryDirectory() as d:
            sintetico.escrever_dfl(p, d, "TESTE")
            self.assertEqual(dfl.jogos(d), ["TESTE"])
            q = dfl.carregar("TESTE", d, usar_cache=False)
        self.assertEqual(len(q.tempos), 2)
        self.assertEqual(q.tempos[1][1] - q.tempos[0][2] - 1, dfl.INTERVALO)
        # as pessoas vêm ordenadas por (lado, PersonId), e "J10" vem antes de "J2"
        ordem = sorted(range(22), key=lambda s: (p.lado[0, s], "J{}".format(s)))
        for (_, a, b), (_, a2, b2) in zip(p.tempos, q.tempos):
            np.testing.assert_array_equal(q.rotulos[a2:b2 + 1], p.rotulos[a:b + 1])
            np.testing.assert_allclose(q.bola[a2:b2 + 1], p.bola[a:b + 1], atol=0.01)
            np.testing.assert_allclose(q.jogadores[a2:b2 + 1], p.jogadores[a:b + 1][:, ordem], atol=0.01,
                                       equal_nan=True)
        np.testing.assert_array_equal(q.lado[0], p.lado[0][ordem])
        r = dfl.inverter(q)
        np.testing.assert_array_equal(r.rotulos == H, q.rotulos == A)
        np.testing.assert_array_equal(r.lado, 1 - q.lado)


class Degradacao(unittest.TestCase):
    def test_fracoes_perto_do_alvo_e_repetivel(self):
        p = sintetico.jogo(900, semente=5)
        q1, par = degradacao.degradar(p, np.random.default_rng(7))
        q2, _ = degradacao.degradar(p, np.random.default_rng(7))
        np.testing.assert_array_equal(np.isnan(q1.bola[:, 0]), np.isnan(q2.bola[:, 0]))
        m = perfil.medir(q1.bola, q1.tempos, p.rotulos)
        self.assertLess(abs(m["falta"]["viva"] - par["falta_viva"]), 0.06)
        self.assertLess(abs(m["falta"]["morta"] - par["falta_morta"]), 0.15)
        self.assertIsNone(q1.altura_bola)
        self.assertTrue(np.isnan(q1.jogadores[..., 0]).sum() > np.isnan(p.jogadores[..., 0]).sum())


class Atributos(unittest.TestCase):
    def test_forma_e_simetria(self):
        p = sintetico.jogo(120, semente=6)
        x, nomes = atributos.matriz(p)
        self.assertEqual(x.shape, (p.n, len(nomes)))
        self.assertEqual(len(set(nomes)), len(nomes))
        # trocar mandante e visitante troca as colunas m e v
        xi, _ = atributos.matriz(dfl.inverter(p))
        j = {n: i for i, n in enumerate(nomes)}
        jogo = p.jogados()
        for a, b in (("perto_1m", "perto_1v"), ("dom_desdem", "dom_desdev"), ("forma_media_xm", "forma_media_xv")):
            np.testing.assert_allclose(x[jogo, j[a]], xi[jogo, j[b]], equal_nan=True, rtol=1e-5)
        np.testing.assert_allclose(x[jogo, j["perto_saldo"]], -xi[jogo, j["perto_saldo"]], equal_nan=True, rtol=1e-5)

    def test_compactar(self):
        pos = np.array([[[1, 1], [np.nan, np.nan], [3, 3], [4, 4]]], dtype=np.float32)
        lado = np.array([[0, 0, 1, 0]], dtype=np.int8)
        p, _ = atributos.compactar(pos, np.zeros_like(pos), lado, 0)
        np.testing.assert_array_equal(p[0], [[1, 1], [4, 4]])


class Modelos(unittest.TestCase):
    def test_proximo_dono(self):
        r = np.array([H, D, D, A, D, X, D, D, H], dtype=np.int8)
        np.testing.assert_array_equal(modelos.proximo_dono(r), [-1, 0, 0, -1, -1, -1, 1, 1, -1])

    def test_treino_salva_e_carrega(self):
        p = sintetico.jogo(300, semente=8)
        x, nomes = atributos.matriz(p)
        ok = p.rotulos != X
        trio = modelos.treinar(x[ok], p.rotulos[ok], modelos.proximo_dono(p.rotulos)[ok],
                               hiper={"max_iter": 20, "min_samples_leaf": 20})
        pm, ph, pr = modelos.prever(trio, x)
        viva = (p.rotulos == H) | (p.rotulos == A)
        self.assertGreater(((ph > 0.5) == (p.rotulos == H))[viva].mean(), 0.8)
        self.assertGreater(((pm > 0.5) == (p.rotulos == D))[ok].mean(), 0.8)
        with tempfile.TemporaryDirectory() as d:
            modelos.salvar(trio, os.path.join(d, "conjuntos", "um"), nomes)
            np.testing.assert_allclose(modelos.probabilidades(x, nomes, d)[1], ph)
            with self.assertRaises(RuntimeError):
                modelos.probabilidades(x[:, :-1], nomes[:-1], d)


class OutrosClassificadores(unittest.TestCase):
    def test_arvores_treinam_salvam_e_carregam(self):
        import importlib.util
        p = sintetico.jogo(240, semente=12)
        x, nomes = atributos.matriz(p)
        ok = p.rotulos != X
        viva = (p.rotulos == H) | (p.rotulos == A)
        pequenos = {"xgboost": {"n_estimators": 20}, "lightgbm": {"n_estimators": 20, "min_child_samples": 20},
                    "catboost": {"iterations": 20}}
        for clf, hiper in pequenos.items():
            if importlib.util.find_spec(clf) is None:
                continue
            with self.subTest(clf=clf):
                trio = modelos.treinar(x[ok], p.rotulos[ok], modelos.proximo_dono(p.rotulos)[ok],
                                       hiper=hiper, classificador=clf)
                pm_, ph, _ = modelos.prever(trio, x)
                self.assertGreater(((ph > 0.5) == (p.rotulos == H))[viva].mean(), 0.8)
                with tempfile.TemporaryDirectory() as d:
                    modelos.salvar(trio, os.path.join(d, "conjuntos", "um"), nomes, classificador=clf)
                    np.testing.assert_allclose(modelos.probabilidades(x, nomes, d)[1], ph, rtol=1e-6)

    def test_redes_treinam_salvam_e_carregam(self):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            self.skipTest("sem PyTorch")
        from posse_ml import sequencia
        p = sintetico.jogo(240, semente=12)
        x, nomes = atributos.matriz(p)
        prox = modelos.proximo_dono(p.rotulos)
        viva = (p.rotulos == H) | (p.rotulos == A)
        with tempfile.TemporaryDirectory() as d:
            caminho = os.path.join(d, "copia.npz")
            config.gravar_npz(caminho, x=x, rot=p.rotulos, prox=prox, tempos=np.array(p.tempos),
                              nomes=np.array(nomes))
            for clf in modelos.SEQUENCIAIS:
                with self.subTest(clf=clf):
                    rede = sequencia.treinar([caminho], clf, hiper={"epocas": 6})
                    pm_, ph, pr = modelos.prever(rede, x, p.tempos)
                    self.assertEqual(len(ph), p.n)
                    self.assertGreater(((ph > 0.5) == (p.rotulos == H))[viva].mean(), 0.7)
                    pasta = os.path.join(d, clf)
                    modelos.salvar(rede, os.path.join(pasta, "conjuntos", "um"), nomes, classificador=clf)
                    np.testing.assert_allclose(modelos.probabilidades(x, nomes, pasta, p.tempos)[1], ph, rtol=1e-5)


class Transmissao(unittest.TestCase):
    def test_acha_os_dois_deslocamentos(self):
        p = sintetico.jogo(900, semente=9)
        rng = np.random.default_rng(0)
        # a transmissão: cada tempo com o seu deslocamento, um replay no meio, bola com ruído
        desl = {1: 4000, 2: 6900}
        n = p.n - 2000
        jog = np.full((n, 22, 2), np.nan, dtype=np.float32)
        b = np.full((n, 2), np.nan, dtype=np.float32)
        for tempo, a, fim in p.tempos:
            for f in range(max(a - desl[tempo], 0), min(fim - desl[tempo] + 1, n)):
                jog[f] = p.jogadores[f + desl[tempo]]
                b[f] = p.bola[f + desl[tempo]] + rng.normal(0, 0.3, 2)
        replay = slice(5000, 5600)
        jog[replay] = jog[replay.start - 3000:replay.stop - 3000]
        b[replay] = b[replay.start - 3000:replay.stop - 3000]
        tra = p.trocar(jogadores=jog, bola=b, lado=p.lado[:n], rotulos=None, tempos=[(1, 0, n - 1)])
        plano, achados = transmissao.planos(p, tra, np.ones(n, dtype=bool))
        self.assertEqual(achados, [4000, 6900])
        self.assertTrue((plano[5000 + 4000:5600 + 4000] == transmissao.OUTRA_VISTA).mean() > 0.9)
        self.assertTrue((plano[1000 + 4000:2000 + 4000] == transmissao.AO_VIVO).mean() > 0.9)


class Perfil(unittest.TestCase):
    def test_lacunas_e_pulos(self):
        n = 20000
        b = np.column_stack([np.linspace(0, 100, n), np.full(n, 30.0)]).astype(np.float32)
        for s in range(1000, 19000, 1000):
            b[s:s + 50] = np.nan
        b[500:503] += [0, 20]
        rot = np.full(n, H, dtype=np.int8)
        m = perfil.medir(b, [(1, 0, n - 1)], rot)
        # as 18 lacunas e o pulo, que o filtro transforma numa lacuna de 3 frames
        self.assertEqual(sum(m["lacunas"]["todas"]), 19)
        self.assertEqual(sum(m["pulos"]["duracao"]), 1)
        self.assertAlmostEqual(m["falta"]["viva"], (18 * 50 + 3) / n, places=4)
        r = np.random.default_rng(0)
        v = perfil.sortear(r, m["lacunas"]["bordas"], m["lacunas"]["todas"], 1000)
        self.assertTrue((((v >= 50) & (v < 75)) | ((v >= 3) & (v < 4))).all())
        self.assertGreater(((v >= 50) & (v < 75)).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
