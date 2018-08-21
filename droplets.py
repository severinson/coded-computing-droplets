'''Code relating to the performance of Liquid MapReduce

- Load simulation
- Mean delay should be such that each server has d droplets. This makes more sense.
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
import typedefs

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
def simulate(f, lmrs, cache=None, rerun=False):
    '''Compute the delay for each lmr in lmrs using f, i.e., f is called
    for each lmr in lmrs. Returns a dataframe.

    '''
    if isinstance(cache, str) and not rerun:
        try:
            return pd.read_csv(cache+'.csv')
        except:
            pass
    logging.info('simulating {} lmrs'.format(len(lmrs)))
    # df = pd.DataFrame([lmr.asdict() for lmr in lmrs])
    df = pd.DataFrame([typedefs.dct_from_lmr(lmr) for lmr in lmrs])
    df['delay'] = pool.map(f, lmrs)
    if isinstance(cache, str):
        df.to_csv(cache+'.csv', index=False)
    return df

def set_wait_for(lmr=None, overhead=None):
    '''Set the optimal number of servers to wait for in-place.

    '''
    def f(q):
        nonlocal lmr, overhead
        lmr['wait_for'] = int(round(q[0]))
        return delay.delay_mean(lmr, overhead=overhead)
        # if q > lmr.nservers:
        #     return q * 1e32
        # if q < 1:
        #     return -(q-2) * 1e32
        # lmr.wait_for = int(math.floor(q))
        # d1 = delay.delay_mean(lmr, overhead=overhead)
        # lmr.wait_for = int(math.ceil(q))
        # d2 = delay.delay_mean(lmr, overhead=overhead)
        # result = d1 * (math.ceil(q)-q) + d2 * (q-math.floor(q))
        # return result

    result = minimize(
        f,
        x0=lmr['nservers']/2,
        bounds=[(1, lmr['nservers'])],
    )
    wait_for = int(result.x.round())
    lmr['wait_for'] = wait_for
    return lmr

def wait_for_optimal(overhead, lmr):
    '''Return the optimal number of servers to wait for.

    '''
    wait_for_orig = lmr.wait_for
    def f(q):
        nonlocal overhead, lmr
        lmr.wait_for = int(round(q[0]))
        return delay.delay_mean(lmr, overhead=overhead)
        # if q > lmr.nservers:
        #     return q * 1e32
        # if q < 1:
        #     return -(q-2) * 1e32
        # lmr.wait_for = int(math.floor(q))
        # d1 = delay.delay_mean(lmr, overhead=overhead)
        # lmr.wait_for = int(math.ceil(q))
        # d2 = delay.delay_mean(lmr, overhead=overhead)
        # result = d1 * (math.ceil(q)-q) + d2 * (q-math.floor(q))
        # return result

    result = minimize(
        f,
        x0=lmr.nservers/2,
        bounds=[(1, lmr.nservers)],
        # method='Powell',
    )
    wait_for = int(result.x.round())
    lmr.wait_for = wait_for_orig
    return wait_for

import unittest
class Tests(unittest.TestCase):

    def lmr1(self):
        return lmr_factory(
            nservers=100,
            nrows=100,
            ncols=100,
            nvectors=100,
            ndroplets=100,
            wait_for=80,
            straggling_factor=1,
            decodingf=lambda x: 0,
        )

    # def lmr1(self):
    #     return LMR(
    #         ndroplets=100,
    #         nservers=100,
    #         nrows=100,
    #         ncols=100,
    #         straggling_factor=100,
    #     )

    # def test_estimates(self):
    #     '''test that the drops/time estimates are each others inverse'''
    #     lmr = self.lmr1()
    #     t1 = 100000
    #     d = drops_estimate(t1, lmr)
    #     self.assertGreater(d, 0)
    #     t2 = delay_estimate(d, lmr)
    #     self.assertAlmostEqual(t1, t2)
    #     return

if __name__ == '__main__':
    unittest.main()
