import unittest
import numpy as np
import stats

class Tests(unittest.TestCase):

    def test_mean(self):
        n = 100
        xp = 0.0
        for k in range(1, n+1):
            x = stats.order_mean_shiftexp(total=n, order=k, parameter=10)
            self.assertGreater(x, xp)
            xp = x
        return

    def test_variance(self):
        n = 20
        xp = 0.0
        for k in range(1, n+1):
            x = stats.order_variance_shiftexp(total=n, order=k, parameter=10)
            self.assertGreater(x, xp)
            xp = x
        return

    def test_cdf(self):
        n = 100
        for k in range(1, n+1):
            self.assertAlmostEqual(
                stats.order_cdf_shiftexp(0.0, total=n, order=k, parameter=10),
                0.0,
            )
            self.assertAlmostEqual(
                stats.order_cdf_shiftexp(1000, total=n, order=k, parameter=10),
                1.0,
            )
            yp = 0.0
            for x in np.linspace(0, 100):
                y = stats.order_cdf_shiftexp(x, total=n, order=k, parameter=10)
                self.assertGreaterEqual(y, yp)
                yp = y
        return
