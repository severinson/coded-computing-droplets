import unittest
import optimize
import complexity
import typedefs

class Tests(unittest.TestCase):

    def lmr1(self):
        lmr = typedefs.lmr_factory(
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

    # def test_to_from_dct(self):
    #     lmr1 = self.lmr1()
    #     dct = typedefs.dct_from_lmr(lmr1)
    #     lmr2 = typedefs.lmr_from_dct(dct)
    #     self.assertEqual(lmr1, lmr2)
    #     return
