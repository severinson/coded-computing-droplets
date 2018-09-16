import math
import numpy as np
import matplotlib.pyplot as plt
import pynumeric
import delay
import stats
import droplets
import complexity
import typedefs
import scipy.integrate as integrate

from functools import partial
from numba import jit, njit, prange
from diskcache import FanoutCache

# cache to disk
cache = FanoutCache('./diskcache')

# pyplot setup
plt.style.use('ggplot')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 300

def lmr1():
    '''return a LMR system'''
    return typedefs.lmr_factory(
        straggling_factor=10000,
        nservers=100,
        nrows=10000,
        ncols=100,
        nvectors=100,
        ndroplets=round(10000*1.5),
        droplet_size=100,
        decodingf=complexity.testf,
    )

def lmr2():
    '''return a LMR system'''
    return typedefs.lmr_factory(
        straggling_factor=10000,
        nservers=300,
        nrows=100000,
        ncols=10,
        nvectors=100,
        ndroplets=round(10000*1.5),
        droplet_size=100,
        decodingf=complexity.testf,
    )

success_dtype = np.dtype([
    ('success', np.int64),
    ('samples', np.int64),
])

@cache.memoize(typed=True)
@njit
def success_from_t(t, d, lmr, n_max=math.inf, success_target=100):
    n = 0
    success = 0
    dropletc = lmr['dropletc']
    a = np.zeros(lmr['nservers'])
    while n-success < success_target and n < n_max:
        a = delay.delays(0, lmr, out=a)
        drops = np.floor(np.maximum(t-a, 0)/dropletc).sum()
        if drops >= d: # and a[-1] <= t:
            success += 1
        n += 1
    return success, n

@cache.memoize(typed=True)
@njit
def success_q_from_t(t, d, lmr, n_max=math.inf, success_target=100):
    n = 0
    success = 0
    dropletc = lmr['dropletc']
    a = np.zeros(lmr['nservers'])
    while n-success < success_target and n < n_max:
        a = delay.delays(0, lmr, out=a)
        drops = np.floor(np.maximum(t-a, 0)/dropletc).sum()
        if drops >= d:
            i = np.searchsorted(a, t, side='right')
            if i >= lmr['wait_for']:
                success += 1
        n += 1
    return success, n

@jit(parallel=True)
def delay_cdf_sim(x, d, lmr):
    '''Evaluate the CDF of the delay at times x.

    Args:

    x: array of delays to evaluate the CDF at.

    d: required number of droplets.

    n: number of samples.

    '''
    cdf = np.zeros(len(x))
    for i in prange(len(x)):
        t = x[i]
        # for i, t in enumerate(x):
        success, samples = success_from_t(t, d, lmr)
        cdf[i] = success/samples
        # print('{}/{}, cdf[i]={}, t={}'.format(i, len(x), cdf[i], t))
        # cdf[i] = result['success']/result['samples']
    return cdf

@jit(parallel=True)
def delay_q_cdf_sim(x, d, lmr):
    '''Evaluate the CDF of the delay at times x.

    Args:

    x: array of delays to evaluate the CDF at.

    d: required number of droplets.

    n: number of samples.

    '''
    cdf = np.zeros(len(x))
    for i in prange(len(x)):
        t = x[i]
        # for i, t in enumerate(x):
        success, samples = success_q_from_t(t, d, lmr)
        cdf[i] = success/samples
        print('{}/{}, cdf[i]={}, t={}'.format(i, len(x), cdf[i], t))
    return cdf

@jit
def cdf_bound(t, d, lmr):
    '''Lower-bound the probability of computing d droplets within time t.

    Considers several events that all mean the computation will
    succeed. The probability of these events overlap. The bound
    consists of finding the most probable of these events.

    '''
    r = 0
    for g in range(1, lmr['nservers']+1):
        T = t - d * lmr['dropletc'] / g
        if T < 0:
            continue
        v = stats.order_cdf_shiftexp(
            T+lmr['straggling'],
            total=lmr['nservers'],
            order=g,
            parameter=lmr['straggling'],
        )
        r = max(r, v)
    return r

@jit
def cdf_q_bound(t, d, lmr):
    '''Lower-bound the probability of computing d droplets and having q
    servers available within time t.

    Considers several events that all mean the computation will
    succeed. The probability of these events overlap. The bound
    consists of finding the most probable of these events.

    '''
    r = 0
    for g in range(1, lmr['nservers']+1):
        T = t - d * lmr['dropletc'] / g
        if T < 0:
            continue
        v = stats.order_cdf_shiftexp(
            T+lmr['straggling'],
            total=lmr['nservers'],
            order=g,
            parameter=lmr['straggling'],
        )
        if g < lmr['wait_for']:
            v *= stats.order_cdf_shiftexp(
                t-T+lmr['straggling'],
                total=lmr['nservers']-g,
                order=lmr['wait_for']-g,
                parameter=lmr['straggling']
            )
        r = max(r, v)
    return r

def plot_cdf(num_droplets=2000):

    # x = np.linspace(0, 100*max(straggling_parameter, complexity), 10)
    # time needed to get the droplets
    lmr = lmr1()
    print(lmr)
    t = delay.delay_estimate(num_droplets, lmr)

    r1 = 1
    r2 = 30
    x1 = np.linspace(t/r2, t*r1, 100)
    x2 = np.linspace(t*r1, t*r2, 100)[:83]
    x = np.concatenate((x1, x2))

    r1_q = 1
    r2_q = 300
    x1_q = np.linspace(t/r2_q, t*r1_q, 100)
    x2_q = np.linspace(t*r1_q, t*r2_q, 100)[:67]
    x_q = np.concatenate((x1_q, x2_q))
    print(x_q)

    # simulated
    cdf = delay_cdf_sim(x, num_droplets, lmr)
    # plt.semilogy(x, 1-cdf, label='simulation')
    plt.loglog(x, 1-cdf, label='Simulation')
    print(cdf)

    cdf_q = delay_q_cdf_sim(x_q, num_droplets, lmr)
    # plt.semilogy(x, 1-cdf, label='simulation')
    plt.loglog(x_q, 1-cdf_q, label='Simulation (q)')
    print(cdf_q)

    # bounds
    cdf = np.fromiter((cdf_bound(t, num_droplets, lmr) for t in x), dtype=float)
    # plt.semilogy(x, 1-cdf, 'k--', label='Upper Bound')
    plt.loglog(x, 1-cdf, 'k--', label='Upper Bound')
    print(cdf)

    cdf_q = np.fromiter((cdf_q_bound(t, num_droplets, lmr) for t in x_q), dtype=float)
    # plt.semilogy(x, 1-cdf, 'k--', label='Upper Bound')
    plt.loglog(x_q, 1-cdf_q, 'k--', label='Upper Bound (q)')
    print(cdf)

    # plt.xlim((1e8, 1e11))
    plt.ylim((1e-7, 1))
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Pr(\rm{Delay} > t$)')
    plt.tight_layout()
    plt.savefig('./plots/180820/bound.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

if __name__ == '__main__':
    plot_cdf()
