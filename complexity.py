'''Decoding complexity for various codes.

'''

import math
import logging

from numba import jit

# arithmetic complexity
WORD_SIZE = 8
ADDITIONC = WORD_SIZE/64
MULTIPLICATIONC = WORD_SIZE*math.log2(WORD_SIZE)

def testf(lmr, reloverhead=1):
    '''Decoding complexity function for tests.

    '''
    assert 1 <= reloverhead < math.inf
    a = 10*lmr['nrows']*reloverhead*ADDITIONC
    m = 10*lmr['nrows']*reloverhead*MULTIPLICATIONC
    return a+m

def decoding0(lmr, reloverhead=1.0):
    return 0

def r10_complexity(lmr, reloverhead=1.02, max_deg=40):
    '''Return the decoding complexity of R10 codes for a given relative
    overhead. Assumes that the number of input symbols K is close to
    1000.

    max_deg gives the maximum value of the degree distribution. 40 is
    the default for R10 codes. Lower values means that the probability
    of all higher degrees have been added to that degree.

    '''
    K = lmr['nrows']/lmr['droplet_size']
    if not 900 <= K <= 1100:
        logging.warning('K={} too far from 1000'.format(K))
        return math.inf
    tbl = {
        (40, 1.02): 137,
        (40, 1.1): 22,
        (11, 1.02): 128,
        (15, 1.02): 130.5,
        (20, 1.02): 132.5,
    }
    if (max_deg, reloverhead) not in tbl:
        raise ValueError('no data for (max_deg, reloverhead)=({},{})'.format(
            max_deg,
            reloverhead,
        ))
    a = tbl[(max_deg, reloverhead)]*K
    a *= lmr['droplet_size']*ADDITIONC
    return a

def lt_complexity(lmr, reloverhead=1.3):
    '''Return the decoding complexity of LT codes for a given relative
    overhead. Assumes that the number of input symbols K is close to
    1000.

    '''
    K = lmr['nrows']/lmr['droplet_size']
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
    a *= lmr['droplet_size']*ADDITIONC
    m *= lmr['droplet_size']*MULTIPLICATIONC
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
    a *= lmr.droplet_size*ADDITIONC
    m *= lmr.droplet_size*MULTIPLICATIONC
    return a+m

def rs_decoding_complexity_fft(lmr, erasure_prob=None, code_length=None):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the FFT-based algorithm in the paper "Novel
    Polynomial Basis With Fast Fourier Transform and its Application
    to Reed-Solomon Erasure Codes".

    Parameters are curve-fit to empiric values.

    Args:

    code_length: The length of the code in number of coded symbols.

    Returns: tuple (additions, multiplications)

    '''
    assert code_length is not None
    f = lambda x, a, b, c: a+b*x*math.log2(c*x)
    a = f(code_length, 2, 8.5, 0.86700826)
    m = f(code_length, 2, 1, 4)
    a *= ADDITIONC
    m *= MULTIPLICATIONC
    return a+m

def rs_decoding_complexity(lmr, erasure_prob=None, code_length=None):
    '''Compute the decoding complexity of Reed-Solomon codes

    Return the number of operations (additions and multiplications)
    required to decode a Reed-Solomon code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm.

    Args:

    code_length: The length of the code in number of coded symbols.

    Returns: tuple (additions, multiplications)

    '''
    assert code_length is not None
    a = code_length * (erasure_prob * code_length - 1) * lmr['droplet_size']
    m = pow(code_length, 2) * erasure_prob * lmr['droplet_size']
    a *= ADDITIONC
    m *= MULTIPLICATIONC
    return a+m

def bdc_decoding_complexity(lmr, erasure_prob=None, code_length=None,
                            partitions=1, reloverhead=1.0, algorithm='bm'):
    '''Compute the decoding complexity of block-diagonal codes

    Return the number of operations (additions and multiplications)
    required to decode a block-diagonal code over the packet erasure
    channel, and using the Berelkamp-Massey algorithm. This function
    considers the asymptotic case as the packet size approaches
    infinity.

    Args:

    code_length: The length of the code in number of coded symbols.

    packet_size: The size of a packet in number of symbols.

    erasure_prob: The erasure probability of the packet erasure channel.

    partitions: The number of block-diagonal code partitions.

    Returns: The total complexity of decoding.

    '''
    if code_length is None:
        code_length = lmr['ndroplets']
    if erasure_prob is None:
        erasure_prob = 1 - lmr['code_rate']
    # assert partitions % 1 == 0
    # assert code_length % partitions == 0, 'Partitions must divide code_length.'
    partition_length = code_length / partitions
    if algorithm == 'bm':
        partition_complexity = rs_decoding_complexity(
            lmr,
            code_length=partition_length,
            erasure_prob=erasure_prob,
        )
    elif algorithm == 'fft':
        partition_complexity = rs_decoding_complexity_fft(
            lmr,
            code_length=partition_length,
            erasure_prob=erasure_prob,
        )
    else:
        raise ValueError('algorithm must be bm or fft')

    return partition_complexity * partitions
