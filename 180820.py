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
from typedefs import lmr_factory

# cache to disk
cache = FanoutCache('./diskcache')

# pyplot setup
plt.style.use('ggplot')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (3, 3)
plt.rcParams['figure.dpi'] = 300

def find_parameters_2(nservers, C=1e4, code_rate=1/3, tol=0.02,
                    ratio=100, straggling_factor=1, wait_for=None):
    '''Get a list of parameters for the size plot.'''

    # assume num_outputs and num_columns is a constant factor of
    # num_source_rows
    # nrows = (pow(ratio, 2)*C*nservers) ** (1./3)
    nrows = math.sqrt(ratio/10*C)

    K_target = 1000 # target number of source symbols
    droplet_size = round(nrows / K_target)
    ndroplets = nrows / droplet_size / code_rate
    ndroplets = round(ndroplets / nservers)*nservers

    # ndroplets = droplets_per_server*nservers
    # droplet_size = round(nrows / code_rate / nservers / droplets_per_server)

    nrows = ndroplets * code_rate * droplet_size
    nvectors = 10*nservers
    ncols = round(nrows/ratio)
    nrows = round(nrows)

    C_emp = nrows*ncols*nvectors/nservers
    err = abs((C-C_emp)/C)
    if err > tol:
        raise ValueError("err={} too big".format(err))

    lmr = lmr_factory(
        nrows=nrows,
        ncols=ncols,
        nvectors=nvectors,
        nservers=nservers,
        straggling_factor=straggling_factor,
        ndroplets=ndroplets,
        droplet_size=droplet_size,
        wait_for=wait_for,
    )
    return lmr

def get_parameters_workload():
    l = list()
    C_target = 1e7
    C0 = C_target
    min_nrows = 0 # ensure nrows is always increasing
    for i in range(20, 1001):
        try:
            lmr = find_parameters_2(
                i, C=C_target,
                straggling_factor=C0/C_target,
                tol=0.1,
                ratio=1000,
            )
        except ValueError as err:
            continue
        if lmr['nrows'] <= min_nrows:
            continue
        min_nrows = lmr['nrows']
        l.append(lmr)
    return l

def get_parameters_straggling():
    C_target = 1e7
    l = list()
    nservers = 625
    for i in np.linspace(1, 5, 20):
        lmr = find_parameters_2(
            nservers, C=C_target,
            straggling_factor=i,
            tol=0.1,
            ratio=1000,
        )
        l.append(lmr)
    return l

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
        print('{}/{}, cdf[i]={}, t={}'.format(i, len(x), cdf[i], t))
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
def single_lower_bound(t, d, g, lmr):
    '''return the probability that g servers become available in time to
    guarantee that d droplets are computed by time t.

    '''
    t1 = g1_deadline(t, d, g, lmr)
    if t1 < 0:
        return 0.0
    result = stats.order_cdf_shiftexp(
        t1+lmr['straggling'],
        total=lmr['nservers'],
        order=g,
        parameter=lmr['straggling'],
    )
    if g < lmr['wait_for']:
        result *= stats.order_cdf_shiftexp(
            t-t1+lmr['straggling'],
            total=lmr['nservers']-g,
            order=lmr['wait_for']-g,
            parameter=lmr['straggling']
        )
    return result

@jit
def cdf_q_bound(t, d=None, lmr=None):
    '''Lower-bound the probability of computing d droplets and having q
    servers available within time t.

    Considers several events that all mean the computation will
    succeed. The probability of these events overlap. The bound
    consists of finding the most probable of these events.

    '''
    result = 0.0
    for g in range(1, lmr['nservers']+1):
        result = max(single_lower_bound(t, d, g, lmr), result)
    return result

@njit
def g1_deadline(t, d, g, lmr):
    '''return the latest time at which server g may become available to
    guarantee that d droplets have been computed by time t.

    '''
    return t - d * lmr['dropletc']/ g

@njit
def g2_deadline(t, t1, d, g1, g2, lmr):
    '''return the latest time at which server g2 (> g1) has to become
    available to guarantee that d droplets have been computed by time
    t. assumes server g1 became available at time t1.

    '''
    return (g2*t - g1*t1 - d*lmr['dropletc']) / (g2-g1)

@jit
def pair_lower_bound(t1, t=None, d=None, g1=None, g2=None, lmr=None):
    '''lower bound the probability that d droplets have been computed and
    wait_for serves has become available by time t.

    '''

    # probability that g1 servers have become available by time t1.
    result = stats.order_pdf_shiftexp(
        t1+lmr['straggling'],
        total=lmr['nservers'],
        order=g1,
        parameter=lmr['straggling']
    )

    # time at which g2 servers has to become available to guarantee
    # that d droplets have been computed by time t.
    t2 = g2_deadline(t, t1, d, g1, g2, lmr)

    # probability that g2 servers has become available at time t2
    # given that g1 servers was available at time t1.
    result *= stats.order_cdf_shiftexp(
        t2-t1+lmr['straggling'],
        total=lmr['nservers']-g1,
        order=g2-g1,
        parameter=lmr['straggling']
    )

    # probability that wait_for servers are available at time t.
    if g2 < lmr['wait_for']:
        result *= stats.order_cdf_shiftexp(
            t-t2+lmr['straggling'],
            total=lmr['nservers']-g2,
            order=lmr['wait_for']-g2,
            parameter=lmr['straggling']
        )

    return result

def bound3_inner(g, t, d, lmr):
    t1 = g1_deadline(t, d, g, lmr)

    # exit if deadline is impossible to meet
    if t1 < 0:
        return 0.0

    # probability that g servers are available in time to guarantee
    # that d droplets have been computed by time t.
    result = stats.order_cdf_shiftexp(
        t1+lmr['straggling'],
        total=lmr['nservers'],
        order=g,
        parameter=lmr['straggling'],
    )

    # we may need to wait for additional servers
    if g < lmr['wait_for']:
        result *= stats.order_cdf_shiftexp(
            t-t1+lmr['straggling'],
            total=lmr['nservers']-g,
            order=lmr['wait_for']-g,
            parameter=lmr['straggling']
        )
    # return result

    # exit if there are no more servers that can become available
    if g == lmr['nservers']:
        return result

    # speed up the computation by exiting here when the probability of
    # success is already within machine epsilon of 0 or 1
    eps = np.finfo(float).eps
    if result <= eps or 1-result <= eps:
        return result

    r_inner = 0
    for g2 in range(g+1, lmr['nservers']+1):
        t2 = g1_deadline(t, d, g2, lmr)
        f = partial(
            pair_lower_bound,
            t=t,
            d=d,
            g1=g,
            g2=g2,
            lmr=lmr,
        )
        v, abserr = integrate.quad(f, t1, t2)
        if np.isnan(v):
            continue
        r_inner = max(r_inner, v)
        # print('b3 inner', t2, v, r_inner, result)

        # exit if combined probability is close to 1
        if (1 - result - r_inner) <= eps:
            break

    result += r_inner

    # deal with floating point round-off error when returning
    return min(result, 1.0)

def bound3(t, d=None, lmr=None):
    result = 0.0
    for g in range(1, lmr['nservers']+1):
        result = max(bound3_inner(g, t, d, lmr), result)
    print('t/result', t, result)
    return result

def mean_delay(lmr, d):
    '''return the mean delay until d droplets have been computed in total
    and lmr['wait_for'] servers have become available.

    '''
    # d_per_vector = lmr['nrows']/lmr['droplet_size']*overhead
    return pynumeric.cnuminv(
        partial(
            cdf_q_bound,
            d=d,
            lmr=lmr,
        ),
        target=1/2,
        tol=1e-6,
    )

def mean_delay3(lmr, d):
    '''return the mean delay until d droplets have been computed in total
    and lmr['wait_for'] servers have become available.

    '''
    return pynumeric.cnuminv(
        partial(
            bound3,
            d=d,
            lmr=lmr,
        ),
        target=1/2,
        tol=None,
        xtol=1e-1,
        ytol=math.inf,
    )

def plot_cdf():
    '''plot the cdf of the following two events:

    - at least d droplets have been computed by time t.

    - at least d droplets have been computed and at least q servers
    have become available by time t.

    '''

    # time needed to get the droplets
    # lmr = lmr1()
    lmr = get_parameters_workload()[-1]
    num_droplets = lmr['nrows']/lmr['droplet_size']*lmr['nvectors']
    t = delay.delay_estimate(num_droplets, lmr)

    r1 = 1
    r2 = 30
    x1 = np.linspace(t/r2, t*r1, 100)
    x2 = np.linspace(t*r1, t*r2, 100)[:1] # 83
    x = np.concatenate((x1, x2))

    r1_q = 1
    r2_q = 300
    x1_q = np.linspace(t/r2_q, t*r1_q, 100)
    x2_q = np.linspace(t*r1_q, t*r2_q, 100)[:1] # 67
    x_q = np.concatenate((x1_q, x2_q))

    # simulated
    cdf = delay_cdf_sim(x, num_droplets, lmr)
    # plt.semilogy(x, 1-cdf, label='simulation')
    plt.loglog(x, 1-cdf, label='Simulation')

    cdf_q = delay_q_cdf_sim(x_q, num_droplets, lmr)
    # plt.semilogy(x, 1-cdf, label='simulation')
    plt.loglog(x_q, 1-cdf_q, label='Simulation (q)')

    # bounds
    x1 = np.linspace(t/r2, t*r1, 100)
    x2 = np.linspace(t*r1, t*r2, 100)
    x = np.concatenate((x1, x2))
    cdf = np.fromiter((cdf_bound(t, num_droplets, lmr) for t in x), dtype=float)
    plt.loglog(x, 1-cdf, 'k-', label='Upper Bound')

    cdf_q = np.fromiter((cdf_q_bound(t, num_droplets, lmr) for t in x), dtype=float)
    plt.loglog(x, 1-cdf_q, 'm--', label='Upper Bound (q)')

    # x = np.linspace(t*r1, t*r2, 2)[:2]
    cdf_3 = np.fromiter((bound3(t, d=num_droplets, lmr=lmr) for t in x), dtype=float)
    plt.loglog(x, 1-cdf_3, 'g:', label='Bound3')
    print(cdf_3)

    # plot average
    # avg = mean_delay(lmr, num_droplets)
    # print('avg', avg)
    # plt.loglog([avg, avg], [cdf_q.max(), cdf_q.min()])

    # plt.xlim((1e8, 1e11))
    plt.ylim((1e-7, 1))
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Pr(\rm{Delay} > t$)')
    plt.tight_layout()
    # plt.savefig('./plots/180820/bound.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def plot_mean():
    '''plot the average delay as computed via bpunding the CDF and via
    function inversion.

    '''
    lmrs = get_parameters_workload()[-3:]
    # lmrs = get_parameters_straggling()
    for lmr in lmrs:
        d = lmr['nrows']/lmr['droplet_size']*lmr['nvectors']
        avg_inv = delay.delay_mean(lmr, overhead=1.0, d_tot=d)
        avg_cdf = mean_delay(lmr, d)
        avg_3 = mean_delay3(lmr, d)
        # avg_3 = 684135424.0
        avg_emp = delay.delay_mean_empiric(lmr, d_tot=d)
        print(lmr)
        print('emp/inv/cdf/3', avg_emp, avg_inv, avg_cdf, avg_3, avg_cdf/avg_inv, avg_3/avg_inv, avg_cdf/avg_emp, avg_3/avg_emp)
        print()

@njit
def mean_available(lmr, out=None):
    '''return an array of length nservers with the average time each
    server becomes available.

    '''
    if out is None:
        out = np.zeros(lmr['nservers'])
    out[0] = stats.order_mean_shiftexp(
        total=lmr['nservers'],
        order=1,
        parameter=lmr['straggling'],
    )
    for i in range(1, lmr['nservers']):
        out[i] = stats.order_mean_shiftexp(
            total=lmr['nservers']-i,
            order=1,
            parameter=lmr['straggling'],
        )
        out[i] += out[i-1]
    return out

def plot_available():
    lmr = get_parameters_workload()[-1]
    av = mean_available(lmr, out=None)
    print(av)
    plt.figure()
    # plt.plot(av, list(range(1, lmr['nservers']+1)), '.', label='cumul.')
    plt.plot(np.diff(av), '.', label='cumul')

    av2 = np.fromiter(
        (stats.order_mean_shiftexp(
            total=lmr['nservers'],
            order=i,
            parameter=lmr['straggling'],
        ) for i in range(1, lmr['nservers']+1)),
        dtype=float,
    )

    # plt.plot(av2, list(range(1, lmr['nservers']+1)), '.', label='order')
    plt.plot(np.diff(av2), '.', label='order')
    plt.legend()

    plt.figure()
    plt.plot(np.diff(av) / np.diff(av2), '.')
    plt.show()

if __name__ == '__main__':
    plot_mean()
