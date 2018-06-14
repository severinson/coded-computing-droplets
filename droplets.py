'''Code relating to the performance of Liquid MapReduce

- Load simulation
- Mean delay should be such that each server has d droplets. This makes more sense.
- DONE: Mean delay assuming perfect knowledge and infinite available computations.
- DONE: Need mean delay assuming limited number of droplets at each server.
- Mean delay assuming randomly selected computations.

'''

import math
import logging
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import pynumeric
import stats
import delay

from multiprocessing import Pool
from scipy.stats import expon
from scipy.optimize import minimize
from numba import njit

class LMR(object):
    '''Liquid MapReduce parameter struct.

    '''
    WORD_SIZE = 8
    ADDITIONC = WORD_SIZE/64
    MULTIPLICATIONC = WORD_SIZE*math.log2(WORD_SIZE)
    def __init__(self, straggling_factor:int=0, decodingf:callable=None, nservers:int=None,
                 nrows:int=None, ncols:int=None, nvectors:int=None,
                 droplet_size:int=1, ndroplets:int=None, wait_for:int=None):
        '''Parameter struct for the Liquid MapReduce system.

        nrows: number of rows of the input matrix.

        ncols: number of columns of the input matrix.

        nvectors: number of input/output vectors. must be divisible by
        the number of servers.

        straggling: scale parameter of the shifted exponential.

        nservers: number of servers.

        droplet_size: number of coded matrix rows in each droplet.

        ndroplets: total number of droplets.

        '''
        if nvectors is None:
            nvectors = nservers
        if wait_for is None:
            wait_for = nservers
        assert 0 <= straggling_factor < math.inf
        assert 0 < nservers < math.inf
        assert 0 < wait_for <= nservers
        assert 0 < nrows < math.inf
        assert 0 < ncols < math.inf
        assert 0 < nvectors < math.inf
        # if nvectors % nservers != 0:
        #     raise ValueError('nvectors must be divisible by nservers')
        if ndroplets % nservers != 0:
            raise ValueError('ndroplets must be divisible by nservers')
        self.nrows = nrows
        self.ndroplets = ndroplets
        self.droplet_size = droplet_size
        if self.code_rate > 1:
            raise ValueError('code rate must be <= 1')
        self.ncols = ncols
        self.nvectors = nvectors
        self.nservers = nservers
        self.straggling_factor = straggling_factor
        self.decodingf = decodingf
        self.wait_for = wait_for
        if wait_for is None:
            self.set_wait_for()
            # self.wait_for = wait_for_optimal(ndroplets*1.1, self)
        return

    def set_wait_for(self, overhead=1.1):
        '''set a good value for wait_for'''
        self.wait_for = wait_for_optimal(1.1, self)
        return

    # @classmethod
    # def init2(cls, nrows:int=None, ncols:int=None, nvectors:int=None,
    #           nservers:int=None, code_rate=None,
    #           droplets_per_server:int=1, straggling_factor:int=0,
    #           decodingf:callable=None):
    #     ndroplets = droplets_per_server*nservers
    #     droplet_size = nrows / code_rate / nservers / droplets_per_server
    #     if droplet_size % 1 != 0:
    #         raise ValueError('droplet_size must be an integer')
    #     return cls(
    #         nrows=nrows,
    #         ncols=ncols,
    #         nvectors=nvectors,
    #         nservers=nservers,
    #         straggling_factor=straggling_factor,
    #         decodingf=decodingf,
    #         ndroplets=ndroplets,
    #         droplet_size=droplet_size,
    #     )

    def workload(self):
        '''Return the number of additions/multiplications computed by each
        server

        '''
        result = self.droplets_per_server * self.droplet_size
        result *= self.ncols*self.nvectors/self.nservers
        return result

    def asdict(self):
        return {
            'nrows': self.nrows,
            'ncols': self.ncols,
            'nvectors': self.nvectors,
            'nservers': self.nservers,
            'ndroplets': self.ndroplets,
            'wait_for': self.wait_for,
            'droplet_size': self.droplet_size,
            'straggling_factor': self.straggling_factor,
        }

    lmr_dtype = np.dtype([
        ('nrows', np.int64),
        ('ncols', np.int64),
        ('nvectors', np.int64),
        ('nservers', np.int64),
        ('ndroplets', np.int64),
        ('wait_for', np.int64),
        ('droplet_size', np.int64),
        ('droplets_per_server', np.float64),
        ('code_rate', np.float64),
        ('straggling', np.float64),
        ('dropletc', np.float64),
        ('decodingc', np.float64),
    ])
    def asdtype(self):
        '''return representation as a numpy dtype'''
        return np.array([(
            self.nrows,
            self.ncols,
            self.nvectors,
            self.nservers,
            self.ndroplets,
            self.wait_for,
            self.droplet_size,
            self.droplets_per_server,
            self.code_rate,
            self.straggling,
            self.dropletc,
            self.decodingc,
        )], dtype=self.lmr_dtype)[0]

    def __repr__(self):
        return str(self.asdict())

    @property
    def droplets_per_server(self):
        return self.ndroplets / self.nservers

    @property
    def dropletc(self):
        a = (self.ncols - 1) * self.ADDITIONC
        m = self.ncols * self.MULTIPLICATIONC
        return (a+m)*self.droplet_size

    @property
    def decodingc(self):
        if self.decodingf is None:
            raise ValueError('decodingf not set')
        return self.decodingf(self) * self.nvectors/self.wait_for

    @property
    def code_rate(self):
        ncrows = self.ndroplets * self.droplet_size
        return self.nrows / ncrows

    @property
    def straggling(self):
        result =  self.nrows*self.ncols*self.nvectors/self.nservers
        result = result * self.ADDITIONC + result * self.MULTIPLICATIONC
        return self.straggling_factor*result

pool = Pool(processes=8)
def simulate(f, lmrs, lmrds=None):
    '''Compute the delay for each lmr in lmrs using f, i.e., f is called
    for each lmr in lmrs. Returns a dataframe.

    '''
    logging.info('simulating {} lmrs'.format(len(lmrs)))
    df = pd.DataFrame([lmr.asdict() for lmr in lmrs])
    if lmrds:
        df['delay'] = pool.map(f, lmrds)
    else:
        df['delay'] = pool.map(f, lmrs)
    return df

def wait_for_optimal(overhead, lmr):
    '''Return the optimal number of servers to wait for.

    '''
    wait_for_orig = lmr.wait_for
    def f(wait_for):
        nonlocal overhead, lmr
        lmr.wait_for = int(round(wait_for[0]))
        return delay.delay_mean(lmr, overhead=overhead)
    result = minimize(f, x0=lmr.nservers/2, bounds=[(1, lmr.nservers)])
    wait_for = int(round(result.x[0]))
    lmr.wait_for = wait_for_orig
    return wait_for

# def wait_for_heuristic(overhead, lmr):
#     '''Heuristic for choosing the number of servers to wait for. Chooses
#     wait_for to be the number of servers that are available on average
#     when all droplets have been computed.

#     '''
#     # return int(round(lmr.nservers*lmr.code_rate))
#     min_wait_for = d_tot / (lmr.droplets_per_server*lmr.nvectors)
#     t = delay.delay_estimate(d_tot, lmr)
#     pdf = delay.server_pdf(t, lmr)
#     cdf = np.cumsum(pdf)
#     wait_for = pynumeric.numinv(lambda i: cdf[i], target=0.1, upper=len(cdf)-1)
#     # f = lambda q: integrate.trapz(pdf[:q]*np.arange(lmr.nservers+1))
#     # wait_for = i
#     wait_for = int(round(wait_for))
#     wait_for = min(max(wait_for, 1), lmr.nservers)
#     # print('wait_for', wait_for, 'nservers', lmr.nservers)
#     return wait_for

import unittest
class Tests(unittest.TestCase):

    def lmr1(self):
        return LMR(
            ndroplets=100,
            nservers=100,
            nrows=100,
            ncols=100,
            straggling_factor=100,
        )

    def test_estimates(self):
        '''test that the drops/time estimates are each others inverse'''
        lmr = self.lmr1()
        t1 = 100000
        d = drops_estimate(t1, lmr)
        self.assertGreater(d, 0)
        t2 = delay_estimate(d, lmr)
        self.assertAlmostEqual(t1, t2)
        return

    def test_server_pdf(self):
        '''test the analytic server pdf against simulations'''
        lmr = self.lmr1()
        t = 100
        pdf_sim = server_pdf_empiric(t, lmr)
        pdf_ana = server_pdf(t, lmr)
        self.assertTrue(np.allclose(pdf_sim, pdf_ana, atol=0.01))
        return

if __name__ == '__main__':
    unittest.main()
