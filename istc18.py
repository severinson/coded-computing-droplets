'''This script generates the plots for our ISTC18 paper.

'''

import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import delay
import complexity
import droplets
import optimize
import bdc

from os import path
from typedefs import lmr_factory
from functools import partial

r10_plot_string = 'b-o'
lt_plot_string = 'm-d'
bdc_plot_string = 'c-^'
centralized_plot_string = 'r-s'
ideal_plot_string = 'k-'
mds_plot_string = 'g:'

def find_parameters(
        nservers,
        C=1e4,
        code_rate=1/3,
        tol=0.02,
        ratio=100,
        straggling_factor=1,
        wait_for=None):
    '''Find a set of valid parameters with the given properties.

    Returns a set of parameters with the given properties. The
    parameters are chosen such that the computation requires C
    multiplications per server (within tol percent). A ValueError is
    raised if no parameters can be found.

    '''

    # assume num_outputs and num_columns is a constant factor of
    # num_source_rows
    nrows = math.sqrt(ratio/10*C)

    K_target = 1000 # target number of source symbols
    droplet_size = round(nrows / K_target)
    ndroplets = nrows / droplet_size / code_rate
    ndroplets = round(ndroplets / nservers)*nservers

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

def get_parameters_straggling():
    '''return a list of parameter structs for the straggling plot.'''
    C_target = 1e7
    l = list()
    nservers = 625
    for i in np.linspace(1, 5, 20):
        lmr = find_parameters(
            nservers,
            C=C_target,
            straggling_factor=i,
            tol=0.1,
            ratio=1000,
        )
        l.append(lmr)
    return l

def get_parameters_workload():
    '''return a list of parameter structs for the workload plot.'''
    l = list()
    C_target = 1e7
    C0 = C_target
    min_nrows = 0 # ensure nrows is always increasing
    for i in range(20, 1001):
        try:
            lmr = find_parameters(
                i,
                C=C_target,
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

def simulate_centralized_r10(lmrs):
    print('R10 cent.')
    for lmr in lmrs:
        decodingf = partial(
            complexity.r10_complexity,
            reloverhead=1.02,
        )
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=decodingf,
        )

    df = droplets.simulate(
        partial(
            delay.delay_mean_centralized,
            overhead=1.020148,
        ),
        lmrs,
    )
    return df

def simulate_centralized_lt(lmrs):
    print('LT cent.')
    for lmr in lmrs:
        decodingf = partial(
            complexity.lt_complexity,
            reloverhead=1.3,
        )
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.3,
            decodingf=decodingf,
        )
    return droplets.simulate(
        partial(
            delay.delay_mean_centralized,
            overhead=1.3,
        ),
        lmrs,
    )

def simulate_ideal(lmrs):
    for lmr in lmrs:
        decodingf = complexity.decoding0
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.0,
            decodingf=decodingf,
            wait_for=1,
        )
    return droplets.simulate(
        partial(delay.delay_mean, overhead=1.0),
        lmrs,
    )

def simulate_mds(lmrs):
    print('MDS')
    for lmr in lmrs:
        decodingf = partial(
            complexity.bdc_decoding_complexity,
            code_length=lmr['nservers'],
            partitions=1,
            algorithm='fft',
        )
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.0,
            decodingf=decodingf,
            wait_for=int(round(lmr['code_rate'] * lmr['nservers']))
        )
    df = droplets.simulate(
        delay.delay_classical,
        lmrs,
    )
    return df

def simulate_lt(lmrs):
    for lmr in lmrs:
        decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.3,
            decodingf=decodingf,
        )
    return droplets.simulate(
        partial(delay.delay_mean, overhead=1.3),
        lmrs,
    )

def simulate_bdc(lmrs, cache=None):
    # BDC
    print('BDC')
    for lmr in lmrs:
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.0,
            decodingf=complexity.decoding0,
            wait_for=1,
        )
    df = droplets.simulate(
        partial(
            delay.delay_mean_simulated,
            overhead=1.0,
            n=100,
            order=delay.ROUND_ROBIN,
        ),
        lmrs,
        cache=cache,
    )

    # add decoding/straggling delay
    for i, lmr in enumerate(lmrs):
        if i >= len(df):
            break
        q, d = bdc.optimize_bdc(lmr)
        tmp = df['delay'][i]
        df.loc[i, 'delay'] += d

    return df

def approximate_r10(lmrs):
    '''Approximation of R10 performance.'''
    print('R10 approximation')
    for lmr in lmrs:
        decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=decodingf,
        )
    r10 = droplets.simulate(
        partial(delay.delay_mean, overhead=1.020148),
        lmrs,
    )
    return r10

def simulate_r10_rr(lmrs, cache=None, rerun=False):
    '''R10 simulated, round-robin droplet order.'''
    print('R10 simulated round-robin')
    for lmr in lmrs:
        decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=decodingf,
        )
    df = droplets.simulate(
        partial(
            delay.delay_mean_simulated,
            overhead=1.020148,
            n=100,
            order=delay.ROUND_ROBIN,
        ),
        lmrs,
        cache=cache,
        rerun=rerun,
    )
    return df

def simulate_r10_opt(lmrs, cache=None):
    '''R10 simulated, optimal droplet order.'''
    for lmr in lmrs:
        decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.02,
            decodingf=decodingf,
            # wait_for=int(round(lmr['nservers']/2))
        )
        print(lmr['wait_for'])
    return droplets.simulate(
        partial(
            delay.delay_mean_empiric,
            overhead=1.020148,
        ),
        lmrs,
        cache=cache,
    )

def straggling_plot():
    cache_dir = './.simulate_cache/'
    cache_prefix = 'workload_'

    # get parameters to simulate
    lmrs = get_parameters_straggling()[:5]

    # run simulations
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)

    mds = simulate_mds(lmrs)
    mds['delay'] /= uncoded['delay']

    bdc = simulate_bdc(
        lmrs,
        cache=path.join(cache_dir, cache_prefix+'bdc'),
    )
    bdc['delay'] /= uncoded['delay']

    r10 = approximate_r10(lmrs)
    r10['delay'] /= uncoded['delay']

    r10_rr = simulate_r10(
        lmrs,
        cache=path.join(cache_dir, cache_prefix+'r10_rr'),
    )
    r10_rr['delay'] /= uncoded['delay']

    # create plot
    plt.figure()
    plt.plot(
        r10['nservers'], r10['delay'],
        r10_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='R10',
    )
    plt.plot(
        r10_rr['nservers'], r10_rr['delay'],
        '--',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='R10 sim. (rr)',
    )
    plt.plot(
        bdc['nservers'], bdc['delay'],
        bdc_plot_string,
        markevery=0.2,
        label='BDC [6]',
    )
    plt.plot(
        mds['nservers'], mds['delay'],
        classical_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='Classical [4]',
    )
    plt.grid(linestyle='--')
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.show()
    return

def workload_plot():
    cache_dir = './.simulate_cache/'
    cache_prefix = 'workload_'

    # get parameters to simulate
    lmrs = get_parameters_workload()[:10]

    # run simulations
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)

    mds = simulate_mds(lmrs)
    mds['delay'] /= uncoded['delay']

    bdc = simulate_bdc(
        lmrs,
        cache=path.join(cache_dir, cache_prefix+'bdc'),
    )
    bdc['delay'] /= uncoded['delay']

    r10 = approximate_r10(lmrs)
    r10['delay'] /= uncoded['delay']

    r10_rr = simulate_r10_rr(
        lmrs,
        cache=path.join(cache_dir, cache_prefix+'r10_rr'),
    )
    r10_rr['delay'] /= uncoded['delay']

    r10_opt = simulate_r10_opt(
        lmrs,
        # cache=path.join(cache_dir, cache_prefix+'r10_rr'),
    )
    r10_opt['delay'] /= uncoded['delay']

    r10_cent = simulate_centralized_r10(lmrs)
    r10_cent['delay'] /= uncoded['delay']

    lt = simulate_lt(lmrs)
    lt['delay'] /= uncoded['delay']

    ideal = simulate_ideal(lmrs)
    ideal['delay'] /= uncoded['delay']

    # create plot
    plt.figure()
    plt.plot(
        r10['nservers'], r10['delay'],
        r10_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='R10',
    )
    plt.plot(
        r10_opt['nservers'], r10_opt['delay'],
        # '--',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='R10 sim. (opt)',
    )
    plt.plot(
        r10_rr['nservers'], r10_rr['delay'],
        '--',
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='R10 sim. (rr)',
    )
    plt.plot(
        lt['nservers'], lt['delay'],
        lt_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='LT',
    )
    plt.plot(
        ideal['nservers'], ideal['delay'],
        ideal_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='Ideal Rateless.',
    )
    plt.plot(
        bdc['nservers'], bdc['delay'],
        bdc_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='BDC [6]',
    )
    plt.plot(
        r10_cent['nservers'], r10_cent['delay'],
        centralized_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='Cent. R10',
    )
    plt.plot(
        mds['nservers'], mds['delay'],
        mds_plot_string,
        markevery=0.2,
        markerfacecolor='none',
        markeredgewidth=1.0,
        label='MDS [4]',
    )
    plt.grid(linestyle='--')
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    workload_plot()
    # straggling_plot()
