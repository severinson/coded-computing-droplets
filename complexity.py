'''Complexity computations for droplet-based schemes.

'''

import math
import logging

def decodingf(lmr):
    a = 10*lmr.nrows*lmr.ADDITIONC
    m = 10*lmr.nrows*lmr.MULTIPLICATIONC
    return a+m

def decoding0(lmr):
    return 0

def r10_complexity(lmr, reloverhead=1.02):
    '''Return the decoding complexity of R10 codes for a given relative
    overhead. Assumes that the number of input symbols K is close to
    1000.

    '''
    K = lmr.nrows/lmr.droplet_size
    if not 900 <= K <= 1100:
        logging.warning('K={} too far from 1000'.format(K))
        return math.inf
    tbl = {
        1.02: 137,
        1.1: 22,
    }
    if reloverhead not in tbl:
        raise ValueError('no data for reloverhead={}'.format(reloverhead))
    a = tbl[reloverhead]*K
    a *= lmr.droplet_size*lmr.ADDITIONC
    return a

def lt_complexity(lmr, reloverhead=1.3):
    '''Return the decoding complexity of LT codes for a given relative
    overhead. Assumes that the number of input symbols K is close to
    1000.

    '''
    K = lmr.nrows/lmr.droplet_size
    if not 900 <= K <= 1100:
        logging.warning('K={} too far from 1000'.format(K))
        return math.inf
    tbl = {
        1.3: 13,
        1.35: 7,
        1.4: 4.3,
    }
    if reloverhead not in tbl:
        raise ValueError('no data for reloverhead={}'.format(reloverhead))
    a = tbl[reloverhead]*K
    m = a
    a *= lmr.droplet_size*lmr.ADDITIONC
    m *= lmr.droplet_size*lmr.MULTIPLICATIONC
    return a+m

def rq_complexity(lmr, reloverhead=1.02):
    '''Return the decoding complexity of RQ codes for a given absolute
    overhead. Assumes that the number of input symbols K is close to
    1000.

    '''
    K = lmr.nrows/lmr.droplet_size
    assert reloverhead == 1.02, 'complexity is only valid for reloverhead=1.02'
    # assert 900 <= K <= 1100, 'complexity is only valid for K close to 1000'
    if not 900 <= K <= 1100:
        logging.warning('K={} too far from 1000'.format(K))
        return math.inf
    a, m = 475*K, 240*K
    a *= lmr.droplet_size*lmr.ADDITIONC
    m *= lmr.droplet_size*lmr.MULTIPLICATIONC
    return a+m
