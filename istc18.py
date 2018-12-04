import math
import matplotlib.pyplot as plt
import delay
import complexity
import droplets
import optimize
import bdc

from typedefs import lmr_factory
from functools import partial

r10_plot_string = 'b-o'
r10d11_plot_string = 'g-v'
rq_plot_string = 'g-^'
lt_plot_string = 'm-d'
bdc_plot_string = 'c-^'
centralized_plot_string = 'r-s'
bound_plot_string = 'k-'
classical_plot_string = 'g:'

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

def raptor_plot():
    lmrs = get_parameters_workload()[:5]
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)

    # MDS
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
    mds = droplets.simulate(
        delay.delay_classical,
        lmrs,
    )
    mds['delay'] /= uncoded['delay']

    # BDC
    for lmr in lmrs:
        optimize.set_wait_for(
            lmr=lmr,
            overhead=1.0,
            decodingf=complexity.decoding0,
            wait_for=1,
        )
    bdcr = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.0, n=100, order='heuristic'),
        lmrs,
        cache='bdc',
    )

    # add decoding/straggling delay
    for i, lmr in enumerate(lmrs):
        if i >= len(bdcr):
            break
        q, d = bdc.optimize_bdc(lmr)
        tmp = bdcr['delay'][i]
        bdcr.loc[i, 'delay'] += d

    bdcr['delay'] /= uncoded['delay']

    # R10
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
    r10['delay'] /= uncoded['delay']

    # R10 simulated, heuristic
    rr = droplets.simulate(
        partial(
            delay.delay_mean_simulated,
            overhead=1.020148,
            n=100,
            order=delay.ROUND_ROBIN,
        ),
        lmrs,
    )
    rr['delay'] /= uncoded['delay']
    print(rr)

    # # R10 simulated, optimal
    # optimal = droplets.simulate(
    #     partial(delay.delay_mean_empiric, overhead=1.020148),
    #     lmrs,
    # )
    # optimal['delay'] /= uncoded['delay']

    # # bound
    # for lmr in lmrs:
    #     lmr.decodingf = complexity.decoding0
    #     lmr.wait_for = 1
    # bound = droplets.simulate(
    #     partial(delay.delay_mean, overhead=1.0),
    #     lmrs,
    # )
    # bound['delay'] /= uncoded['delay']

    plt.figure()
    plt.plot(
        r10['nservers'], r10['delay'], r10_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10')
    plt.plot(
        rr['nservers'], rr['delay'], '--', markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10 sim. (rr)')
    plt.plot(mds['nservers'], mds['delay'])
    plt.plot(
        bdcr['nservers'], bdcr['delay'], bdc_plot_string, markevery=0.2,
        label='BDC [6]')
    plt.plot(
        mds['nservers'], mds['delay'], classical_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='Classical [4]')
    plt.grid(linestyle='--')
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.xlim(0, 600)
    plt.ylim(0.2, 0.7)
    plt.show()
    return

if __name__ == '__main__':
    raptor_plot()
