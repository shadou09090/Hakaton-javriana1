"""Regresiones de geometría que afectan las etiquetas de entrenamiento."""
import unittest
from preparar_datos import etiquetas_recorte, posiciones


class RecortesTest(unittest.TestCase):
    def test_imagen_menor_normaliza_dimension_real(self):
        self.assertEqual(posiciones(481, 512, 0.2), [0])
        rows = etiquetas_recorte([[0, 0.5, 0.5, 0.4, 0.4]], 481, 411, 0, 0, 512)
        for actual, expected in zip(rows[0], [0, .5, .5, .4, .4]):
            self.assertAlmostEqual(actual, expected)

    def test_bordes_unicos(self):
        self.assertEqual(posiciones(820, 512, .2), [0, 308])
        self.assertEqual(posiciones(1024, 512, .2), [0, 409, 512])

    def test_fragmento_rechaza_ventana(self):
        self.assertIsNone(etiquetas_recorte([[1, .5, .5, .8, .8]], 1024, 1024, 0, 0, 512))

    def test_recorte_remapea_caja(self):
        rows = etiquetas_recorte([[4, .75, .75, .2, .2]], 1024, 1024, 512, 512, 512)
        for actual, expected in zip(rows[0], [4, .5, .5, .4, .4]):
            self.assertAlmostEqual(actual, expected)

    def test_fondo_es_vacio(self):
        self.assertEqual(etiquetas_recorte([[0, .1, .1, .1, .1]], 1024, 1024, 512, 512, 512), [])


if __name__ == "__main__":
    unittest.main()
