'''Delay computations for droplet-bases schemes.

'''

import math
import random
import numpy as np
import pynumeric
import stats
import scipy.special.lambertw

from scipy.special import lambertw
from numba import jit, njit

def delay_uncoded(lmr):
    '''Return the delay of the uncoded system.

    '''
    rows_per_server = lmr.nrows/lmr.nservers
    result = rows_per_server*(lmr.ncols-1)*lmr.ADDITIONC
    result += rows_per_server*lmr.ncols*lmr.MULTIPLICATIONC
    result *= lmr.nvectors
    result += stats.ShiftexpOrder(
        parameter=lmr.straggling,
        total=lmr.nservers,
        order=lmr.nservers,
    ).mean()
    return result

def delay_classical(lmr):
    '''Return the delay of the classical scheme of Lee based on MDS codes
    and no partial computations.

    '''
    # delay due to the computation
    rows_per_server = lmr.nrows/lmr.nservers/lmr.code_rate
    result = rows_per_server*(lmr.ncols-1)*lmr.ADDITIONC
    result += rows_per_server*lmr.ncols*lmr.MULTIPLICATIONC
    result *= lmr.nvectors

    # delay due to waiting for the q-th server
    result += stats.ShiftexpOrder(
        parameter=lmr.straggling,
        total=lmr.nservers,
        order=lmr.wait_for,
    ).mean()

    # delay due to decoding
    result += lmr.decodingc

    return result

def delay_mean_empiric(lmr, overhead=1.1, d_tot=None, n=100):
    '''Return the simulated mean delay of the map phase.

    Assumes that the map phase ends whenever a total of d droplets
    have been computed and the slowest server is available. Also
    assumes that which droplet to compute next is selected optimally.

    '''
    if d_tot is None:
        d_tot = lmr.nrows*lmr.nvectors/lmr.droplet_size*overhead
    result = 0
    max_drops = lmr.droplets_per_server*lmr.nvectors
    if max_drops * lmr.nservers < d_tot:
        return math.inf
    dropletc = lmr.dropletc # cache this value
    for _ in range(n):
        a = delays(None, lmr)
        t_servers = a[lmr.wait_for-1] # a is sorted
        f = lambda x: np.floor(
            np.minimum(np.maximum((x-a)/dropletc, 0), max_drops)
        ).sum()
        t_droplets = pynumeric.cnuminv(f, d_tot, tol=dropletc)
        result += max(t_servers, t_droplets)
    return result/n + lmr.decodingc

def delay_mean_centralized(lmr, overhead=1.1):
    '''Return the mean delay when there is a central reducer, i.e., a
    single master node that does all decoding.

    '''
    d_per_vector = lmr.nrows/lmr.droplet_size*overhead
    t_decoding = delay_estimate(d_per_vector, lmr)
    t_decoding += lmr.decodingc*lmr.wait_for
    t_droplets = delay_estimate(d_per_vector*lmr.nvectors, lmr)
    t_droplets += lmr.decodingc*lmr.wait_for/lmr.nvectors
    # print('t_decoding/t_droplets', t_decoding, t_droplets, t_decoding/t_droplets, lmr.wait_for)
    return max(t_decoding, t_droplets)

@njit
def arg_from_order(droplet_order, nvectors, d):
    '''Return the index of droplet_order at which the map phase ends.

    Args:

    droplet_order: array indicating which order the droplets are
    computed in.

    d: required number of droplets per vector.

    '''
    i = 0
    droplets_by_vector = np.zeros(nvectors)
    for v in droplet_order:
        droplets_by_vector[v] += 1
        if droplets_by_vector.min() >= d:
            break
        i += 1
    return i

def delay_mean_simulated(lmr, overhead=1.1, n=10, order='random'):
    '''Return the simulated mean delay of the map phase.

    Assumes that the map phase ends whenever a total of d droplets
    have been computed and the slowest server is available. Which
    droplet to compute is chosen randomly.

    order: strategy for choosing the optimal droplet order. must be
    either 'random' or 'heuristic'.

    '''
    assert order in ['random', 'heuristic']
    d_tot = lmr.nrows*lmr.nvectors/lmr.droplet_size*overhead
    max_drops = lmr.droplets_per_server*lmr.nvectors
    if max_drops * lmr.nservers < d_tot:
        return math.inf

    result = 0
    nsamples = int(math.ceil(lmr.nservers*max_drops))
    droplet_order = np.zeros(nsamples, dtype=int)
    t = np.zeros(nsamples)
    dropletc = lmr.dropletc # cache this value
    v = np.zeros(lmr.nvectors)
    v[:] = np.arange(lmr.nvectors)
    server_droplet_order = np.tile( # this is the heuristic order
        np.arange(lmr.nvectors),
        int(lmr.droplets_per_server),
    )
    for k in range(n):
        a = delays(None, lmr)
        assert len(a) == lmr.nservers
        print('simulating order={} {}/{}'.format(order, k, n))
        for i in range(lmr.nservers):
            if order == 'random':
                np.random.shuffle(server_droplet_order)
            elif order == 'heuristic':
                j = random.randint(0, len(v))
                # j = i % lmr.nvectors
                v[:j] = np.arange(len(v)-j, len(v))
                v[j:] = np.arange(len(v)-j)
                # np.random.shuffle(v)
                server_droplet_order[:] = np.tile(v, int(lmr.droplets_per_server))
                # server_droplet_order[:j], server_droplet_order[j:] = server_droplet_order[-j-1:], server_droplet_order[:-j-1]
            else:
                raise ValueError('order must be random or heuristic')
            j1 = int(i*lmr.nvectors*lmr.droplets_per_server)
            j2 = int((i+1)*lmr.nvectors*lmr.droplets_per_server)
            droplet_order[j1:j2] = server_droplet_order[:]
            t[j1:j2] = a[i] + dropletc*np.arange(
                1,
                lmr.nvectors*lmr.droplets_per_server+1,
            )
        p = np.argsort(t)
        droplet_order = droplet_order[p]
        t = t[p]
        i = arg_from_order(droplet_order, lmr.nvectors, d_tot/lmr.nvectors)
        if i >= len(t):
            return math.inf
        t_droplets = t[i]
        t_servers = a[lmr.wait_for-1] # a is sorted
        result += max(t_servers, t_droplets)

    return result/n + lmr.decodingc

# @jit
def delay_mean(lmr, overhead=1.0, d_tot=None):
    '''Return the mean delay of the map phase when d_tot droplets are
    required in total. If d_tot is not given it's computed from the
    overhead. The returned value is an upper bound on the true mean.

    '''
    if d_tot is None:
        d_tot = lmr['nrows']*lmr['nvectors']/lmr['droplet_size']*overhead
    max_drops = lmr['droplets_per_server']*lmr['nvectors']
    if max_drops * lmr['nservers'] < d_tot:
        return math.inf
    t = delay_estimate(d_tot, lmr)
    result = t
    pdf = server_pdf(t, lmr)
    for i in range(lmr['wait_for']):
        rv = stats.ShiftexpOrder(
            parameter=lmr['straggling'],
            total=lmr['nservers']-i,
            order=lmr['wait_for']-i,
        )
        result += pdf[i] * (rv.mean() - lmr['straggling'])
    return result + lmr['decodingc']

def drops_estimate(t, lmr):
    '''Return an approximation of the number of droplets computed at time
    t. The inverse of time_estimate.

    '''
    v = -2*lmr.straggling
    v -= lmr.dropletc
    v += (2*lmr.straggling+lmr.dropletc)*math.exp(-t/lmr.straggling)
    v += 2*t
    v /= 2*lmr.dropletc
    return max(lmr.nservers*v, 0)

def drops_lower(t, lmr):
    '''Return a lower bound on the average number of droplets computed at
    time t.

    '''
    v = lmr.straggling
    v += lmr.dropletc
    v -= t
    v -= (lmr.straggling+lmr.dropletc)*math.exp(-t/lmr.straggling)
    v /= lmr.dropletc
    v *= -1
    return max(lmr.nservers*v, 0)

def drops_upper(t, lmr):
    '''Return an upper bound on the average number of droplets computed at
    time t.

    '''
    v = lmr.straggling * (math.exp(-t/lmr.straggling)-1)
    v += t
    v /= lmr.dropletc
    return max(lmr.nservers*v, 0)

def drops_empiric(t, lmr, n=100):
    '''Return the average number of droplets computed at time t. Result
    due to simulations.

    '''
    result = 0
    max_drops = lmr.droplets_per_server*lmr.nvectors
    # max_drops = math.inf
    dropletc = lmr.dropletc # cache this value
    for _ in range(n):
        a = delays(t, lmr)
        result += np.floor(
            np.minimum(np.maximum((t-a)/dropletc, 0), max_drops)
        ).sum()
    return result/n

@jit
def delay_estimate(d_tot, lmr):
    '''Return an approximation of the delay t at which d droplets have
    been computed in total over the K servers. The inverse of
    drops_estimate.

    '''
    t = lmr['straggling'] + lmr['dropletc']/2 + d_tot*lmr['dropletc']/lmr['nservers']
    earg = (lmr['nservers']+2*d_tot)*lmr['dropletc']
    earg /= -2*lmr['nservers']*lmr['straggling']
    earg -= 1
    Warg = math.exp(earg)
    Warg *= 2*lmr['straggling'] + lmr['dropletc']
    Warg /= -2*lmr['straggling']
    t += lmr['straggling'] * lambertw(Warg)
    # t += lmr.straggling * scipy.special.lambertw(Warg)
    return np.real(t)

def delay_lower(d, lmr):
    '''Return a lower bound on the average amount of time required to
    compute d droplets.

    '''
    t = lmr.straggling
    t += d*lmr.dropletc/lmr.nservers
    earg = d*lmr.dropletc
    earg /= -lmr.straggling*lmr.nservers
    earg -= 1
    Warg = -math.exp(earg)
    t += lmr.straggling*lambertw(Warg)
    return np.real(t)

def delay_upper(d, lmr):
    '''Return an upper bound on the average amount of time required to
    compute d droplets.

    '''
    t = lmr.straggling
    t += d*lmr.dropletc/lmr.nservers
    t += lmr.dropletc
    earg = d*lmr.dropletc
    earg += lmr.nservers*(lmr.straggling+lmr.dropletc)
    earg /= -lmr.nservers*lmr.straggling
    Warg = math.exp(earg) * (lmr.straggling+lmr.dropletc)
    Warg /= -lmr.straggling
    t += lmr.straggling * lambertw(Warg)
    return np.real(t)

def delay_lower2(d, lmr):
    '''Lower bound on the average delay required to compute d
    droplets. Based on solving for t and averaging over the initial
    delay.

    '''
    return (d*lmr.dropletc + lmr.straggling) / lmr.nservers

def delay_upper2(d, lmr):
    '''Upper bound on the average delay required to compute d
    droplets. Based on solving for t and averaging over the initial
    delay.

    '''
    v = d/lmr.nservers + 1
    v *= lmr.dropletc
    v += lmr.straggling
    return v

def delay_lower3(d, lmr):
    '''Lower bound on the average delay required to compute d
    droplets. Takes into account the non-linearity.

    '''
    pdf = server_pdf(t, lmr)

def delay_estimate_error(t, lmr):
    '''Return an upper bound on the average error of delay_estimate.

    '''
    return lmr.nservers * (1-math.exp(-t/lmr.straggling))/2

def delays(t, lmr):
    '''Return an array of length <= nservers with simulated delays <= t.

    '''
    a = np.random.exponential(
        scale=lmr.straggling,
        size=lmr.nservers,
    )
    a.sort()
    if t:
        i = np.searchsorted(a, t)
        a = a[:i]
    return a

def server_pdf_empiric(t, lmr, n=100000):
    '''Return the PDF over the number of servers with a delay <= t.
    Computed via simulations.

    '''
    pdf = np.zeros(lmr.nservers+1)
    for _ in range(n):
        i = len(delays(t, lmr))
        pdf[i] += 1
    pdf /= n
    return pdf

@jit
def server_pdf(t, lmr):
    '''Return the PDF over the number of servers with a delay less than t,
    i.e., pdf[0] is the probability that exactly 0 servers have a
    delay at most t, pdf[10] is the probability that 10 servers have a
    delay at most t etc.

    '''
    pdf = np.zeros(lmr['nservers']+1)
    pdf[0] = 1-stats.ShiftexpOrder(
        parameter=lmr['straggling'],
        order=1,
        total=lmr['nservers'],
    ).cdf(t+lmr['straggling'])
    for i in range(1, lmr['nservers']+1):
        rv1 = stats.ShiftexpOrder(
            parameter=lmr['straggling'],
            order=i,
            total=lmr['nservers'],
        )
        if i < lmr['nservers']:
            rv2 = stats.ShiftexpOrder(
                parameter=lmr['straggling'],
                order=i+1,
                total=lmr['nservers'],
            )
            pdf[i] = rv1.cdf(t+lmr['straggling'])
            pdf[i] -= rv2.cdf(t+lmr['straggling'])
        else:
            pdf[i] = rv1.cdf(t+lmr['straggling'])

    return pdf
