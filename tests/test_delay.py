import unittest
import delay
import optimize
import complexity
import numpy as np

from typedefs import lmr_factory

class Tests(unittest.TestCase):

    def lmr1(self):
        lmr = lmr_factory(
            nservers=100,
            nrows=100,
            ncols=100,
            nvectors=100,
            ndroplets=200,
            wait_for=80,
            straggling_factor=1,
            decodingf=lambda x: 0,
        )
        lmr = optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=complexity.testf,
            wait_for=50,
        )
        return lmr

    def test_server_pdf_1(self):
        lmr = self.lmr1()
        t = 10.00
        pdf = delay.server_pdf(t, lmr)
        self.assertAlmostEqual(pdf.sum(), 1.0)
        self.assertGreaterEqual(pdf.min(), 0)
        return

    def test_server_pdf_2(self):
        '''test the analytic server pdf against simulations'''
        lmr = self.lmr1()
        t = 100
        pdf_sim = delay.server_pdf_empiric(t, lmr)
        pdf_ana = delay.server_pdf(t, lmr)
        self.assertTrue(np.allclose(pdf_sim, pdf_ana, atol=0.01))
        return

    def test_delay(self):
        lmr = self.lmr1()
        t = delay.delay_mean(lmr)
        self.assertAlmostEqual(t, 494646.0133995558)
        return

    def test_optimize(self):
        lmr = self.lmr1()
        t1 = delay.delay_mean(lmr)
        lmr = optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=complexity.testf,
        )
        t2 = delay.delay_mean(lmr)

        # make sure optimization didn't increase the delay
        self.assertLessEqual(t2, t1)
        return

    def test_delay(self):
        lmr = self.lmr1()
        delays = delay.delays(100, lmr)
        delays = delay.delays(0, lmr)
        return

    def test_simulate_delay(self):
        lmr = self.lmr1()
        t_random = delay.delay_mean_simulated(lmr, order=0)
        t_rr = delay.delay_mean_simulated(lmr, order=1)
        t_ana = delay.delay_mean(lmr)
        print(t_random, t_rr, t_ana)
        self.assertGreaterEqual(t_random, t_rr)
        self.assertGreaterEqual(t_rr, t_ana)
        return

if __name__ == '__main__':
    unittest.main()
