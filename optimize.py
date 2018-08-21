import math
import delay

from scipy.optimize import minimize

def set_wait_for(lmr=None, overhead=None, decodingf=None, wait_for=None):
    '''Set the number of servers to wait for in the map phase. Also
    computes and stores the decoding delay.

    Args:

    overhead: relative reception overhead. for example, 1.1 for a 10%
    reception overhead.

    decodingf: function that takes arguments (lmr, reloverhead) and
    returns the decoding complexity per vector.

    wait_for: number of servers to wait for. if None, it is optimized
    to minimize the overall delay.

    '''
    def f(q):
        nonlocal lmr, overhead, decodingf
        if q > lmr['nservers']:
            return q * 1e32
        if q < 1:
            return -(q-2) * 1e32
        set_wait_for(
            lmr=lmr,
            overhead=overhead,
            decodingf=decodingf,
            wait_for=int(math.floor(q[0])),
        )
        d1 = delay.delay_mean(lmr, overhead=overhead)
        set_wait_for(
            lmr=lmr,
            overhead=overhead,
            decodingf=decodingf,
            wait_for=int(math.ceil(q[0])),
        )
        d2 = delay.delay_mean(lmr, overhead=overhead)
        result = d1 * (math.ceil(q)-q) + d2 * (q-math.floor(q))
        return result

    if wait_for is None:
        result = minimize(
            f,
            x0=lmr['nservers']/2,
            # bounds=[(1, lmr['nservers'])],
            method='Powell',
        )
        wait_for = int(result.x.round())

    lmr['wait_for'] = wait_for
    lmr['decodingc'] = decodingf(lmr, reloverhead=overhead)
    lmr['decodingc'] *= lmr['nvectors'] / lmr['wait_for']
    return lmr
