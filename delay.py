'''Delay computations for droplet-bases schemes.

'''

import math
import random
import numpy as np
import stats
import typedefs
import complexity

from pynverse import inversefunc
from scipy.special import lambertw
from numba import jit, njit, prange

def delay_uncoded(lmr):
    '''Return the delay of the uncoded system.

    '''
    rows_per_server = lmr['nrows']/lmr['nservers']
    result = rows_per_server*(lmr['ncols']-1)*complexity.ADDITIONC
    result += rows_per_server*lmr['ncols']*complexity.MULTIPLICATIONC
    result *= lmr['nvectors']
    result += stats.ShiftexpOrder(
        parameter=lmr['straggling'],
        total=lmr['nservers'],
        order=lmr['nservers'],
    ).mean()
    return result

@njit
def delay_classical(lmr):
    '''Return the delay of the classical scheme of Lee based on MDS codes
    and no partial computations.

    '''
    # delay due to the computation
    rows_per_server = lmr['nrows']/lmr['nservers']/lmr['code_rate']
    result = rows_per_server*(lmr['ncols']-1)*complexity.ADDITIONC
    result += rows_per_server*lmr['ncols']*complexity.MULTIPLICATIONC
    result *= lmr['nvectors']

    # delay due to waiting for the q-th server
    result += stats.order_mean_shiftexp(
        parameter=lmr['straggling'],
        total=lmr['nservers'],
        order=lmr['wait_for'],
    )

    # delay due to decoding
    result += lmr['decodingc']

    return result

def delay_mean_empiric(lmr, overhead=1.1, d_tot=None, n=100):
    '''return the simulated mean delay of the map phase, i.e., the average
    delay until d droplets have been computed and wait_for servers
    have become available. assumes that droplets are computed in
    optimal order.

    '''
    if d_tot is None:
        d_tot = lmr['nrows']*lmr['nvectors']/lmr['droplet_size']*overhead
    result = 0.0
    max_drops = lmr['droplets_per_server']*lmr['nvectors']
    if max_drops * lmr['nservers'] < d_tot:
        return math.inf
    dropletc = lmr['dropletc'] # cache this value
    a = np.zeros(lmr['nservers'])
    for _ in range(n):
        delays(0, lmr, out=a)
        t_servers = a[lmr['wait_for']-1] # a is sorted
        f = lambda x: np.floor(
            np.minimum(np.maximum((x-a)/dropletc, 0), max_drops)
        ).sum()
        t_droplets = inversefunc(f, y_values=d_tot)[0]
        result += max(t_servers, t_droplets)

    # average and add decoding delay
    result /= n
    result += lmr['decodingc']
    return result

def delay_mean_centralized(lmr, overhead=1.1):
    '''Return the mean delay when there is a central reducer, i.e., a
    single master node that does all decoding.

    '''
    d_per_vector = lmr['nrows']/lmr['droplet_size']*overhead
    t_decoding = delay_estimate(d_per_vector, lmr)
    t_decoding += lmr['decodingc']*lmr['wait_for']
    t_droplets = delay_estimate(d_per_vector*lmr['nvectors'], lmr)
    t_droplets += lmr['decodingc']*lmr['wait_for']/lmr['nvectors']
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

RANDOM = 0
ROUND_ROBIN = 1
@njit
def delay_mean_simulated(lmr, overhead=1.1, n=10, order=ROUND_ROBIN):
    '''Return the simulated mean delay of the map phase.

    Assumes that the map phase ends whenever a total of d droplets
    have been computed and the slowest server is available. Which
    droplet to compute is chosen randomly.

    order: strategy for choosing the optimal droplet order. 0 for
    random order and 1 for round-robin order.

    '''
    assert order in [0, 1]
    d_tot = lmr.nrows*lmr.nvectors/lmr.droplet_size*overhead
    max_drops = lmr.droplets_per_server*lmr.nvectors
    if max_drops * lmr.nservers < d_tot:
        return math.inf

    result = 0
    nsamples = int(math.ceil(lmr.nservers*max_drops))
    droplet_order = np.zeros(nsamples, dtype=np.int64)
    t = np.zeros(nsamples)
    dropletc = lmr.dropletc # cache this value
    v = np.zeros(lmr.nvectors)
    v[:] = np.arange(lmr.nvectors)
    server_droplet_order = np.zeros(
        int(lmr.nvectors*lmr.droplets_per_server),
        dtype=np.int64,
    )
    for i in range(lmr.droplets_per_server):
        server_droplet_order[i*lmr.nvectors:(i+1)*lmr.nvectors] = v

    a = np.zeros(lmr['nservers'])
    for k in range(n):
        a = delays(0, lmr, a)
        assert len(a) == lmr.nservers
        for i in range(lmr.nservers):
            if order == RANDOM:
                np.random.shuffle(server_droplet_order)
            elif order == ROUND_ROBIN:
                j = random.randint(0, len(v))
                v[:j] = np.arange(len(v)-j, len(v))
                v[j:] = np.arange(len(v)-j)
                for j in range(lmr.droplets_per_server):
                    server_droplet_order[j*lmr.nvectors:(j+1)*lmr.nvectors] = v
            else:
                raise ValueError('order must be 0 (random) or 1 (round-robin)')
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
            print('return inf 2')
            return math.inf
        t_droplets = t[i]
        t_servers = a[lmr.wait_for-1] # a is sorted
        result += max(t_servers, t_droplets)
        print(k, '/', n, 'simulations done')

    return result/n + lmr.decodingc

# @jit
def delay_mean(lmr, overhead=1.0, d_tot=0):
    '''Return the mean delay of the map phase when d_tot droplets are
    required in total. If d_tot is zero it's computed from the
    overhead. The returned value is an upper bound on the true mean.

    '''
    if d_tot == 0:
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

@jit(forceobj=True)
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
    return np.real(t)

@njit
def delays(t, lmr, out=None):
    '''Return an array of length <= nservers with simulated delays <=
    t. if t=0, all delays are kept.

    '''
    if out is None:
        out = np.zeros(lmr['nservers'])
    for i in prange(lmr['nservers']):
        out[i] = np.random.exponential(scale=lmr['straggling'])
    out.sort()
    return out

@njit
def server_pdf_empiric(t, lmr, n=100000):
    '''Return the simulated PDF over the number of servers with a delay of
    at most t.

    '''
    pdf = np.zeros(lmr['nservers']+1)
    for _ in prange(n):
        i = len(delays(t, lmr))
        pdf[i] += 1
    pdf /= n
    return pdf

@jit(forceobj=True)
def server_pdf(t, lmr):
    '''Return the PDF over the number of servers with a delay less than t,
    i.e., pdf[0] is the probability that exactly 0 servers have a
    delay at most t, pdf[10] is the probability that 10 servers have a
    delay at most t etc.

    '''
    pdf = np.zeros(lmr['nservers']+1, dtype=np.float64)
    pdf[0] = 1-stats.order_cdf_shiftexp(
        t+lmr['straggling'],
        total=lmr['nservers'],
        order=1,
        parameter=lmr['straggling'],
    )
    pdf[lmr['nservers']] = stats.order_cdf_shiftexp(
        t+lmr['straggling'],
        total=lmr['nservers'],
        order=lmr['nservers'],
        parameter=lmr['straggling'],
    )
    for i in range(1, lmr['nservers']):
        pdf[i] = stats.order_cdf_shiftexp(
            t+lmr['straggling'],
            total=lmr['nservers'],
            order=i,
            parameter=lmr['straggling'],
        )
        pdf[i] -= stats.order_cdf_shiftexp(
            t+lmr['straggling'],
            total=lmr['nservers'],
            order=i+1,
            parameter=lmr['straggling'],
        )
    np.clip(pdf, 0, None, out=pdf) # clip values to (0, inf)
    return pdf
