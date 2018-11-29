'''This module defines a custom numpy dtype describing a coded
computing system. Specifically, it describes a struct containing all
parameters necessery for describing the system. Many functions in this
project take a struct of this kind as its argument.

'''

import math
import numpy as np

# arithmetic complexity
WORD_SIZE = 8
ADDITIONC = WORD_SIZE/64
MULTIPLICATIONC = WORD_SIZE*math.log2(WORD_SIZE)

# dtype containing system parameters
lmr_dtype = np.dtype([
    ('nrows', np.int64),
    ('ncols', np.int64),
    ('nvectors', np.int64),
    ('nservers', np.int64),
    ('ndroplets', np.int64),
    ('wait_for', np.int64),
    ('droplet_size', np.int64),
    ('droplets_per_server', np.float64),
    ('code_rate', np.float64),
    ('straggling', np.float64),
    ('dropletc', np.float64),
    ('decodingc', np.float64),
])

def lmr_factory(nservers:int=None,
                nrows:int=None,
                ncols:int=None,
                nvectors:int=None,
                droplet_size:int=1,
                ndroplets:int=None,
                wait_for:int=None,
                straggling_factor:int=0,
                decodingf:callable=None):
    '''Return an lmr_dtype object containing the given system
    parameters. Call optimize.set_wait_for to set the number of
    servers to wait for after creating the lmr object.

    '''
    assert nservers is not None
    assert nrows is not None
    assert ncols is not None
    assert nvectors is not None
    assert droplet_size is not None
    assert ndroplets is not None
    assert straggling_factor is not None
    if wait_for is None:
        wait_for = int(round(nservers/2))

    droplets_per_server = ndroplets / nservers
    a = (ncols - 1) * ADDITIONC
    m = ncols * MULTIPLICATIONC
    dropletc = (a+m)*droplet_size
    ncrows = ndroplets * droplet_size
    code_rate = nrows / ncrows
    straggling = nrows*ncols*nvectors/nservers
    straggling *= ADDITIONC + MULTIPLICATIONC
    straggling *= straggling_factor
    decodingc = 0 # computed after instantitation
    lmr = np.array([(
        nrows,
        ncols,
        nvectors,
        nservers,
        ndroplets,
        wait_for,
        droplet_size,
        droplets_per_server,
        code_rate,
        straggling,
        dropletc,
        decodingc,
    )], dtype=lmr_dtype)[0]
    if decodingf is not None:
        lmr['decodingc'] = decodingf(lmr) * nvectors/wait_for
    return lmr

def dct_from_lmr(lmr):
    '''Convert a lmr to a dict.

    '''
    return {
        'nrows': lmr['nrows'],
        'ncols': lmr['ncols'],
        'nvectors': lmr['nvectors'],
        'nservers': lmr['nservers'],
        'ndroplets': lmr['ndroplets'],
        'wait_for': lmr['wait_for'],
        'droplet_size': lmr['droplet_size'],
        'droplets_per_server': lmr['droplets_per_server'],
        'code_rate': lmr['code_rate'],
        'straggling': lmr['straggling'],
        'dropletc': lmr['dropletc'],
        'decodingc': lmr['decodingc'],
    }

# def lmr_from_dct(dct):
#     '''Convert a dict to an lmr object.

#     '''
#     return lmr_factory(
#         nservers=dct['nservers'],
#         nrows=dct['nrows'],
#         ncols=dct['ncols'],
#         nvectors=dct['nvectors'],
#         droplet_size=dct['droplet_size'],
#         ndroplets=dct['ndroplets'],
#         wait_for=dct['wait_for'],
#         straggling_factor=dct['straggling_factor'],
#     )
