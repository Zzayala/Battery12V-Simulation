import sys
import math
from unittest.mock import MagicMock

# 1. Mock dependencies BEFORE importing the module
# Mock numpy
class FakeNumpy:
    def exp(self, x):
        try:
            return math.exp(x)
        except TypeError:
            return [math.exp(i) for i in x]
    def array(self, x): return list(x) if hasattr(x, '__iter__') else x
    def mean(self, x): return sum(x)/len(x) if hasattr(x, '__len__') and len(x) > 0 else 0
    def clip(self, a, a_min, a_max):
        if hasattr(a, '__iter__'):
             return [max(min(x, a_max), a_min) for x in a]
        return max(min(a, a_max), a_min)
    def linspace(self, start, stop, num, endpoint=True):
        if num < 1: return []
        if num == 1: return [start]
        step = (stop - start) / (num - 1) if endpoint else (stop - start) / num
        return [start + step*i for i in range(num)]
    def full(self, shape, fill_value):
        if isinstance(shape, int): return [fill_value] * shape
        return [fill_value] * shape[0]
    def zeros(self, shape):
        if isinstance(shape, int): return [0.0] * shape
        return [0.0] * shape[0]
    def all(self, x): return all(x)
    # Functions used in execution block but not in utility functions tests can be loose
    def max(self, x): return max(x) if x else 0
    def min(self, x): return min(x) if x else 0
    def diff(self, x): return [x[i+1]-x[i] for i in range(len(x)-1)]
    def where(self, x): return [i for i, v in enumerate(x) if v]
    def concatenate(self, x): return [item for sublist in x for item in sublist]
    def any(self, x): return any(x)
    def sqrt(self, x): return math.sqrt(x)
    def cumsum(self, x):
        res = []
        s = 0
        for v in x:
            s += v
            res.append(s)
        return res

fake_np = FakeNumpy()
sys.modules['numpy'] = fake_np

# Mock matplotlib
mock_plt = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = mock_plt

mock_widgets = MagicMock()
mock_widgets.Slider = MagicMock
mock_widgets.Button = MagicMock
mock_widgets.RadioButtons = MagicMock
sys.modules['matplotlib.widgets'] = mock_widgets

mock_ticker = MagicMock()
mock_ticker.FuncFormatter = MagicMock
sys.modules['matplotlib.ticker'] = mock_ticker

# Mock scipy
mock_scipy = MagicMock()
mock_scipy_interp = MagicMock()
# interp1d returns a function
def fake_interp1d(x, y, **kwargs):
    def interpolator(val):
        # simple linear interpolation for scalar val
        # assume x is sorted
        if val <= x[0]: return y[0]
        if val >= x[-1]: return y[-1]
        for i in range(len(x)-1):
            if x[i] <= val <= x[i+1]:
                t = (val - x[i]) / (x[i+1] - x[i])
                return y[i] + t * (y[i+1] - y[i])
        return y[0]
    return interpolator

mock_scipy_interp.interp1d = fake_interp1d
sys.modules['scipy'] = mock_scipy
sys.modules['scipy.interpolate'] = mock_scipy_interp

# 2. Import the module under test
import unittest
from unittest.mock import patch, mock_open
import os

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Simulador_Degradacion12V as sim

class TestSimuladorDegradacion(unittest.TestCase):

    def test_get_absolute_path(self):
        filename = "test_file.txt"
        result = sim.get_absolute_path(filename)
        self.assertTrue(result.endswith(filename))

    def test_time_formatter(self):
        self.assertEqual(sim.time_formatter(0, None), "00:00")
        self.assertEqual(sim.time_formatter(65, None), "01:05")
        self.assertEqual(sim.time_formatter(3600, None), "60:00")

    def test_calcular_resistencia_dinamica(self):
        r_base = 0.01

        # Case 1: Standard
        res = sim.calcular_resistencia_dinamica(r_base, 50.0, 25.0, 0.0, 0.0)
        # We need to calculate expected value using our fake numpy (which uses math)
        # R_soc = 0.01 * 1.0 = 0.01
        # Arrhenius: exp(1500 * (1/298.15 - 1/298.15)) = exp(0) = 1.0
        # Butler-Volmer: 1.0 - 0.15*(1 - exp(0)) = 1.0
        # Time: 1.0 + 0.3*(1 - exp(0)) = 1.0
        # Total = 0.01
        self.assertAlmostEqual(res, 0.01, places=5)

        # Case 2: Low SOC (<10)
        # SOC 0 -> val = 1.0 -> factor_soc = 1.5
        res = sim.calcular_resistencia_dinamica(r_base, 0.0, 25.0, 0.0, 0.0)
        self.assertAlmostEqual(res, 0.015, places=5)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="0, 3.0\n50, 3.6\n100, 4.2")
    def test_cargar_curva_ocv_usuario_success(self, mock_file, mock_exists):
        mock_exists.return_value = True

        f_ocv = sim.cargar_curva_ocv_usuario("dummy.txt")

        # Test interpolation using our fake interp1d
        self.assertAlmostEqual(f_ocv(0), 3.0)
        self.assertAlmostEqual(f_ocv(50), 3.6)
        self.assertAlmostEqual(f_ocv(100), 4.2)
        self.assertAlmostEqual(f_ocv(25), 3.3)

    @patch('os.path.exists')
    def test_cargar_curva_ocv_usuario_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        f_ocv = sim.cargar_curva_ocv_usuario("missing.txt")
        # Should return default linear 3.0 to 4.2
        self.assertAlmostEqual(f_ocv(0), 3.0)
        self.assertAlmostEqual(f_ocv(100), 4.2)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data="0, 10, 5.0\n10, 20, -5.0")
    def test_cargar_perfil_solicitaciones_success(self, mock_file, mock_exists):
        mock_exists.return_value = True

        t, I = sim.cargar_perfil_solicitaciones("dummy.txt")

        # Our fake linspace returns lists
        self.assertTrue(len(t) > 0)
        self.assertEqual(len(t), len(I))
        self.assertEqual(I[0], 5.0)
        self.assertEqual(I[-1], -5.0)

    @patch('os.path.exists')
    def test_cargar_perfil_solicitaciones_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        t, I = sim.cargar_perfil_solicitaciones("missing.txt")
        # Should return default (1800 steps, zeros)
        self.assertEqual(len(t), 1800)
        self.assertTrue(all(v == 0 for v in I))

if __name__ == '__main__':
    unittest.main()
