'''Code for simulating the performance of the system.

'''

import math
import logging
import numpy as np
import pandas as pd
import delay
import typedefs

from multiprocessing import Pool
from scipy.stats import expon
from scipy.optimize import minimize

def simulate(f, lmrs, cache=None, rerun=False):
    '''Compute the delay for each lmr in lmrs using f, i.e., f is called
    for each lmr in lmrs. Returns a dataframe.

    if cache is a string, the results of the simulation are cached in
    the file cache+'.csv'. if rerun is False and the file already
    exists its contents are returned instead of running the
    simulation.

    '''
    if isinstance(cache, str) and not rerun:
        try:
            filename = cache+'.csv'
            df = pd.read_csv(filename)
            logging.info(f'loaded cache {filename}')
            return df
        except:
            pass
    logging.info('simulating {} lmrs'.format(len(lmrs)))
    df = pd.DataFrame([typedefs.dct_from_lmr(lmr) for lmr in lmrs])
    df['delay'] = list(map(f, lmrs))
    if isinstance(cache, str):
        filename = cache+'.csv'
        df.to_csv(filename, index=False)
        logging.info(f'caching {filename}')
    return df

def set_wait_for(lmr=None, overhead=None):
    '''Set the optimal number of servers to wait for in-place.

    '''
    def f(q):
        nonlocal lmr, overhead
        lmr['wait_for'] = int(round(q[0]))
        return delay.delay_mean(lmr, overhead=overhead)

    result = minimize(
        f,
        x0=lmr['nservers']/2,
        bounds=[(1, lmr['nservers'])],
    )
    wait_for = int(result.x.round())
    lmr['wait_for'] = wait_for
    return lmr

import unittest
class Tests(unittest.TestCase):

    def lmr1(self):
        return typedefs.lmr_factory(
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
