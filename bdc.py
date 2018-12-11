'''Code relating to the block-diagonal coding scheme.

'''

import math
import stats
import complexity

from scipy.optimize import minimize

def optimize_q_bdc(lmr, num_partitions):
    '''Optimize the value of q for the BDC scheme

    '''
    def f(q):
        nonlocal lmr
        q = q[0]

        # enforce bounds
        if q > lmr['nservers']:
            return q * 1e32
        if q < 1:
            return -(q-2) * 1e32

        decoding = complexity.bdc_decoding_complexity(
            lmr,
            partitions=num_partitions,
        )
        decoding *= lmr['nvectors'] / q

        # interpolate between the floor and ceil of q
        s1 = stats.ShiftexpOrder(
            parameter=lmr['straggling'],
            total=lmr['nservers'],
            order=int(math.floor(q)),
        ).mean()
        s2 = stats.ShiftexpOrder(
            parameter=lmr['straggling'],
            total=lmr['nservers'],
            order=int(math.ceil(q)),
        ).mean()
        straggling = s1 * (math.ceil(q)-q) + s2 * (q-math.floor(q))
        return decoding + straggling

    result = minimize(
        f,
        x0=lmr['nservers']-1,
        method='Powell',
    )
    wait_for = int(result.x.round())
    delay = result.fun

    # add the overhead due to partitioning
    # this is an upper bound
    delay += lmr['dropletc'] * num_partitions / 2

    return wait_for, delay

def optimize_bdc(lmr):
    '''Optimize the number of partitions and the number of servers to wait
    for.

    '''
    min_q = None
    min_d = math.inf
    min_T = None
    max_T = int(round(lmr['nrows'] / lmr['droplet_size']))
    for T in range(int(round(max_T*0.9)), max_T+1):
        q, d = optimize_q_bdc(lmr, T)
        if d < min_d:
            min_d = d
            min_q = q
            min_T = T

    return min_q, min_d
