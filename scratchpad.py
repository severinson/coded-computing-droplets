import math
import logging
import numpy as np
import matplotlib.pyplot as plt
import stats
import scipy.integrate as integrate
import pynumeric
import droplets
import delay
import complexity

from matplotlib2tikz import save as tikz_save
from functools import partial
from scipy.stats import expon
from scipy.special import lambertw
from numba import njit

# pyplot setup
plt.style.use('seaborn-paper')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (3, 3)
plt.rcParams['figure.dpi'] = 300

def estimate_waste():
    a = get_delay(verbose=False)
    d = np.floor((t-a)/complexity)
    assert d.min() >= 0
    waste = t - a - d*complexity
    est = complexity/2
    print('mean waste is {}. est is {}'.format(waste.mean(), est))
    print('{} droplets computed'.format(d.sum()))
    plt.hist(waste,bins=50)
    plt.show()
    return

def estimate_hist():
    # d, est1, est2 = estimates()
    samples = np.fromiter((estimates()[0] for _ in range(1000)), dtype=int)
    est1 = np.fromiter((estimates()[1] for _ in range(1000)), dtype=int)
    est2 = np.fromiter((estimates()[2] for _ in range(1000)), dtype=int)
    print('true mean is', samples.mean())
    # print('est1 mean is', est1.mean())
    print('est2 mean is', est2.mean())

    plt.figure()
    plt.hist(samples, label='true')
    # plt.hist(est1, label='est1')
    plt.hist(est2, label='est2')
    plt.legend()
    plt.show()
    return

def delay_samples(t):
    samples = list()
    lsamples = list()
    for _ in range(10000):
        a = get_delay(t)
        samples.append(a.sum())
        lsamples.append(len(a))
    return np.fromiter(samples, dtype=float), np.fromiter(lsamples, dtype=float)

def delay_hist():
    # it's probably right that the exp rvs just scale. but the gamma
    # does something else. we should maybe just settle for the
    # asymptotic expression. it does pretty well..
    t = 0.5*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=0.5')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 1*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=1')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 2*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=2')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 3*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=3')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    t = 4*straggling_parameter
    samples, lsamples = delay_samples(t)
    plt.hist(samples, bins=50, density=True, label='thr=4')
    print(lsamples.mean()*straggling_parameter, samples.mean())

    plt.legend()
    plt.grid()
    plt.show()
    return

def plot_integrand():
    t = 100*straggling_parameter
    bf = lambda x: math.floor((t-x) / complexity)
    bf2 = lambda x: (t-x) / complexity
    f = lambda x: expon.pdf(x, scale=straggling_parameter)*bf(x)
    x = np.linspace(0, t, 10)
    y = [bf(i) for i in x]
    # plt.plot(x, y, label='floor')
    y2 = [bf2(i) for i in x]
    # plt.plot(x, y2, label='no floor')
    plt.plot(x, [i2-i1 for i1, i2 in zip(y, y2)], label='diff')

    dx1 = straggling_parameter+complexity
    dy1 = bf2(dx1) - bf(dx1)
    a1 = dy1/dx1
    f = lambda x: a1*x
    x = np.linspace(0, straggling_parameter+complexity)
    plt.plot(x, [f(i) for i in x], '--')

    dx2 = t
    dy2 = 0
    a2 = -dy1 / (dx2-dx1)
    c = dy1 - a2*dx1
    f = lambda x: a2*x+c
    x = np.linspace(straggling_parameter+complexity, t)
    plt.plot(x, [f(i) for i in x], '--')

    plt.grid()
    plt.legend()
    plt.show()
    return

def exptest():
    '''Plots to investigate if an exponential RV truncated on the left is
    a shifted exponential.

    '''
    trunc = 2
    norm = 1-expon.cdf(trunc, scale=straggling_parameter)
    print(norm)
    t = np.linspace(0, trunc)
    tt = np.linspace(trunc, 2*trunc)
    plt.plot(t, expon.pdf(tt, scale=straggling_parameter)/norm, label='shifted')
    plt.plot(t, expon.pdf(t, scale=straggling_parameter), label='orig')
    plt.grid()
    plt.legend()
    plt.show()
    return

def lmr1():
    '''return a LMR system'''
    return droplets.LMR(
        straggling_factor=10000,
        nservers=100,
        nrows=10000,
        ncols=100,
        ndroplets=round(10000*1.5),
        droplet_size=100,
        decodingf=complexity.testf,
    )

def plot_mean_delay():
    '''Plot the empiric and analytic mean delay.

    '''
    lmr = lmr1()
    d0 = lmr.nrows * lmr.nvectors
    # d1 = round(d0*0.9)
    d1 = 1
    d2 = round(d0*1.3)
    d = np.linspace(d1, d2, 10, dtype=int)
    sim = [droplets.delay_mean_empiric(i, lmr) for i in d]
    print('got simulations (optimal)')
    # sim_random = [droplets.delay_mean_empiric_random(i, lmr) for i in d]
    # print('got simulations (random)')
    ana = [droplets.delay_mean(i, lmr) for i in d]
    print('got analytic mean')
    # ana_centralized = [droplets.delay_mean_centralized(i, lmr) for i in d]
    # print('got analytic mean (centralized)')
    plt.plot(d, sim, '-o', markevery=0.2, label='simulated (optimal)')
    # plt.plot(d, sim_random, '--', label='simulated (random)')
    plt.plot(d, ana, label='analytic')
    # plt.plot(d, ana_centralized, '-s', markevery=0.2, label='centralized')
    plt.grid()
    plt.legend()
    plt.show()
    return

def find_parameters(nservers, C=1e4, code_rate=1/3, tol=0.02,
                    ratio=100, straggling_factor=1, wait_for=None):
    '''Get a list of parameters for the size plot.'''

    # assume num_outputs and num_columns is a constant factor of
    # num_source_rows
    nrows = (pow(ratio, 2)*C*nservers) ** (1./3)

    K_target = 1000 # target number of source symbols
    droplet_size = round(nrows / K_target)
    ndroplets = nrows / droplet_size / code_rate
    ndroplets = round(ndroplets / nservers)*nservers

    # ndroplets = droplets_per_server*nservers
    # droplet_size = round(nrows / code_rate / nservers / droplets_per_server)

    nrows = ndroplets * code_rate * droplet_size
    # nvectors = round(nrows/ratio / nservers) * nservers
    nvectors = round(nrows/ratio)
    ncols = nvectors
    nrows = round(nrows)

    C_emp = nrows*ncols*nvectors/nservers
    err = abs((C-C_emp)/C)
    if err > tol:
        raise ValueError("err={} too big".format(err))

    lmr = droplets.LMR(
        nrows=nrows,
        ncols=ncols,
        nvectors=nvectors,
        nservers=nservers,
        straggling_factor=straggling_factor,
        decodingf=complexity.testf,
        ndroplets=ndroplets,
        droplet_size=droplet_size,
        wait_for=wait_for,
    )
    return lmr

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

    lmr = droplets.LMR(
        nrows=nrows,
        ncols=ncols,
        nvectors=nvectors,
        nservers=nservers,
        straggling_factor=straggling_factor,
        decodingf=complexity.testf,
        ndroplets=ndroplets,
        droplet_size=droplet_size,
        wait_for=wait_for,
    )
    return lmr

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
        if lmr.nrows <= min_nrows:
            continue
        min_nrows = lmr.nrows
        l.append(lmr)
    return l

def get_parameters_q():
    C_target = 1e8
    l = list()
    nservers = 500
    for q in np.linspace(1, 500, 100, dtype=int):
        lmr = find_parameters(
            nservers=nservers,
            C=C_target,
            wait_for=q,
            ratio=100,
        )
        l.append(lmr)
    return l

def straggling_plot():
    '''Plot delay for different straggling parameters.

    '''
    lmrs = get_parameters_straggling()
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)

    # classical
    for lmr in lmrs:
        lmr.decodingf = partial(
            complexity.bdc_decoding_complexity,
            code_length=lmr.nservers,
            partitions=1,
            algorithm='fft',
        )
        lmr.wait_for = int(round(lmr.code_rate * lmr.nservers))
    classical = droplets.simulate(
        delay.delay_classical,
        lmrs,
    )
    classical['delay'] /= uncoded['delay']

    # BDC
    for lmr in lmrs:
        lmr.decodingf = complexity.decoding0
        lmr.wait_for = 1
    bdc = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.0, n=100, order='heuristic'),
        lmrs,
        cache='bdc_straggling_upper',
    )

    # add decoding/straggling delay
    for i, lmr in enumerate(lmrs):
        q, d = optimize_bdc(lmr)
        tmp = bdc['delay'][i]
        bdc.loc[i, 'delay'] += d

    bdc['delay'] /= uncoded['delay']

    # bound
    for lmr in lmrs:
        lmr.decodingf = complexity.decoding0
        lmr.wait_for = 1
    bound = droplets.simulate(
        partial(delay.delay_mean, overhead=1.0),
        lmrs,
    )

    bound['delay'] /= uncoded['delay']

    # centralized
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.set_wait_for()
    centralized = droplets.simulate(
        partial(delay.delay_mean_centralized, overhead=1.020148),
        lmrs,
    )
    centralized['delay'] /= uncoded['delay']

    # centralized LT
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
        lmr.set_wait_for()
    clt = droplets.simulate(
        partial(delay.delay_mean_centralized, overhead=1.3),
        lmrs,
    )
    clt['delay'] /= uncoded['delay']

    # RQ
    # for lmr in lmrs:
    #     lmr.decodingf = partial(complexity.rq_complexity, reloverhead=1.02)
    #     lmr.set_wait_for()
    # rq = droplets.simulate(
    #     partial(delay.delay_mean, overhead=1.02),
    #     lmrs,
    # )
    # rq['delay'] /= uncoded['delay']

    # R10
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.set_wait_for()
    r10 = droplets.simulate(
        partial(delay.delay_mean, overhead=1.020148),
        lmrs,
    )
    r10['delay'] /= uncoded['delay']

    # R10_d11
    # for lmr in lmrs:
    #     lmr.decodingf = partial(complexity.r10_complexity, max_deg=11, reloverhead=1.02)
    #     lmr.set_wait_for()
    # r10d11 = droplets.simulate(
    #     partial(delay.delay_mean, overhead=1.020148),
    #     lmrs,
    # )
    # r10d11['delay'] /= uncoded['delay']

    # LT
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
        lmr.set_wait_for()
    lt = droplets.simulate(
        partial(delay.delay_mean, overhead=1.3),
        lmrs,
    )
    lt['delay'] /= uncoded['delay']

    # make plot
    plt.figure()
    plt.plot(
        r10['straggling_factor'], r10['delay'],
        r10_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10')
    # plt.plot(r10d11['straggling_factor'], r10d11['delay'], r10d11_plot_string, markevery=0.2, label='R10d11')
    # plt.plot(rq['straggling_factor'], rq['delay'], rq_plot_string, markevery=0.2, label='RQ')
    plt.plot(
        lt['straggling_factor'], lt['delay'], lt_plot_string, markevery=0.3,
        markerfacecolor='none', markeredgewidth=1.0, label='LT')
    plt.plot(
        bdc['straggling_factor'], bdc['delay'], bdc_plot_string, markevery=0.25,
        label='BDC [6]')
    plt.plot(
        centralized['straggling_factor'], centralized['delay'],
        centralized_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10 cent.')
    # plt.plot(clt['straggling_factor'], clt['delay'],
    #          '-', markevery=0.2, label='LT cent. [9]')
    plt.plot(
        classical['straggling_factor'], classical['delay'],
        classical_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='Classical [4]')
    plt.plot(
        bound['straggling_factor'], bound['delay'], bound_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='Ideal')

    # random = droplets.simulate(delay.delay_mean_random, lmrs)
    # random['delay'] /= uncoded['delay']
    # centralized = droplets.simulate(delay.delay_mean_centralized, lmrs)
    # centralized['delay'] /= uncoded['delay']
    # simulated = droplets.simulate(delay.delay_mean_empiric, lmrs)
    # simulated['delay'] /= uncoded['delay']
    # analytic = droplets.simulate(delay.delay_mean, lmrs)
    # analytic['delay'] /= uncoded['delay']
    # plt.figure()
    # plt.plot(
    #     simulated['straggling_factor'], simulated['delay'],
    #     '-o', markevery=0.2, label='Simulated (optimal)',
    # )
    # plt.plot(random['straggling_factor'], random['delay'],
    #          '--', label='Simulated (random)')
    # plt.plot(analytic['straggling_factor'], analytic['delay'], label='Analytic')
    # plt.plot(centralized['straggling_factor'], centralized['delay'],
    #          '-s', markevery=0.2, label='Centralized')

    plt.grid(linestyle='--')
    plt.ylabel(r'$D$')
    plt.xlabel(r'$\beta/\sigma_{\mathsf{K}}$')
    plt.xlim(1, 5)
    plt.ylim(0, 1.6)
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    tikz_save(
        './plots/istc18/straggling.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth',
    )
    plt.savefig('./plots/istc18/straggling.pdf', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def estimate_plot():
    '''Plot delay assuming a constant workload per server.

    '''
    lmrs = get_parameters_workload()
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.set_wait_for()

    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)
    heuristic = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.020148, n=100, order='heuristic'),
        lmrs,
        cache='heuristic',
    )
    heuristic['delay'] /= uncoded['delay']
    random = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.020148, n=100, order='random'),
        lmrs,
        cache='random',
    )
    random['delay'] /= uncoded['delay']
    # centralized = droplets.simulate(delay.delay_mean_centralized, lmrs)
    # centralized['delay'] /= uncoded['delay']
    simulated = droplets.simulate(
        partial(delay.delay_mean_empiric, overhead=1.020148),
        lmrs,
    )
    simulated['delay'] /= uncoded['delay']
    analytic = droplets.simulate(
        partial(delay.delay_mean, overhead=1.020148),
        lmrs,
    )
    analytic['delay'] /= uncoded['delay']
    plt.figure()
    plt.plot(simulated['nservers'], simulated['delay'],
             r10_plot_string, markevery=0.2, label='Simulated (optimal)')
    plt.plot(random['nservers'], random['delay'],
             'g-^', markevery=0.2, label='Simulated (random)')
    plt.plot(heuristic['nservers'], heuristic['delay'],
             'r-s', markevery=0.3, label='Simulated (round-robin)')
    plt.plot(analytic['nservers'], analytic['delay'], 'kd--', markevery=0.25, label='Analytic')
    # plt.plot(
    #     simulated['nservers'], simulated['delay'],
    #     '-o', markevery=0.2, label='Simulated (optimal)',
    # )
    # plt.plot(analytic['nservers'], analytic['delay'], label='Analytic')
    # plt.plot(centralized['nservers'], centralized['delay'],
    #          '-s', markevery=0.2, label='Centralized')
    plt.grid(linestyle='--')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.xlim(0, 600)
    plt.ylim(0.2, 0.36)
    plt.legend(framealpha=1)
    plt.savefig('./plots/istc18/estimate.pdf', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def bounds_plot():
    lmr = lmr1()
    d1 = 1
    d2 = 1000
    d = np.linspace(d1, d2, dtype=int)
    avg = [delay.delay_estimate(x, lmr) for x in d]
    lower = [delay.delay_lower(x, lmr) for x in d]
    upper = [delay.delay_upper(x, lmr) for x in d]
    lower2 = [delay.delay_lower2(x, lmr) for x in d]
    upper2 = [delay.delay_upper2(x, lmr) for x in d]
    plt.plot(d, avg, label='avg.')
    plt.plot(d, lower, label='lower')
    plt.plot(d, upper, label='upper')
    plt.plot(d, lower2, label='lower2')
    plt.plot(d, upper2, label='upper2')
    plt.grid()
    plt.legend()
    plt.show()
    return

r10_plot_string = 'b-o'
r10d11_plot_string = 'g-v'
rq_plot_string = 'g-^'
lt_plot_string = 'm-d'
bdc_plot_string = 'c-^'
centralized_plot_string = 'r-s'
bound_plot_string = 'k-'
classical_plot_string = 'g:'

def raptor_plot():
    lmrs = get_parameters_workload()
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)

    # classical
    for lmr in lmrs:
        lmr.decodingf = partial(
            complexity.bdc_decoding_complexity,
            code_length=lmr.nservers,
            partitions=1,
            algorithm='fft',
        )
        lmr.wait_for = int(round(lmr.code_rate * lmr.nservers))
    classical = droplets.simulate(
        delay.delay_classical,
        lmrs,
    )
    classical['delay'] /= uncoded['delay']

    # centralized LT
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
        lmr.set_wait_for()
    clt = droplets.simulate(
        partial(delay.delay_mean_centralized, overhead=1.3),
        lmrs,
    )
    clt['delay'] /= uncoded['delay']

    # BDC
    for lmr in lmrs:
        lmr.decodingf = complexity.decoding0
        lmr.wait_for = 1
    bdc = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.0, n=100, order='heuristic'),
        lmrs,
        cache='bdc_upper',
    )

    # add decoding/straggling delay
    for i, lmr in enumerate(lmrs):
        if i >= len(bdc):
            continue
        q, d = optimize_bdc(lmr)
        tmp = bdc['delay'][i]
        bdc.loc[i, 'delay'] += d
        # print(lmr)
        # print(q, d, tmp, tmp / bdc['delay'][i])
        # print()

    bdc['delay'] /= uncoded['delay']

    # centralized R10
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.set_wait_for()
    centralized = droplets.simulate(
        partial(delay.delay_mean_centralized, overhead=1.020148),
        lmrs,
    )
    centralized['delay'] /= uncoded['delay']

    # RQ
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.rq_complexity, reloverhead=1.02)
        lmr.set_wait_for()
    rq = droplets.simulate(
        partial(delay.delay_mean, overhead=1.02),
        lmrs,
    )
    rq['delay'] /= uncoded['delay']

    # R10
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.set_wait_for()
    r10 = droplets.simulate(
        partial(delay.delay_mean, overhead=1.020148),
        lmrs,
    )
    r10['delay'] /= uncoded['delay']

    # R10 simulated, heuristic
    rr = droplets.simulate(
        partial(delay.delay_mean_simulated, overhead=1.020148, n=100, order='heuristic'),
        lmrs,
        cache='heuristic_upper',
    )
    rr['delay'] /= uncoded['delay']

    # R10 simulated, optimal
    optimal = droplets.simulate(
        partial(delay.delay_mean_empiric, overhead=1.020148),
        lmrs,
    )
    optimal['delay'] /= uncoded['delay']

    # bound
    for lmr in lmrs:
        lmr.decodingf = complexity.decoding0
        lmr.wait_for = 1
    bound = droplets.simulate(
        partial(delay.delay_mean, overhead=1.0),
        lmrs,
    )
    bound['delay'] /= uncoded['delay']

    # R10_d11
    # for lmr in lmrs:
    #     lmr.decodingf = partial(complexity.r10_complexity, max_deg=11, reloverhead=1.02)
    #     lmr.set_wait_for()
    # r10d11 = droplets.simulate(
    #     partial(delay.delay_mean, overhead=1.020148),
    #     lmrs,
    # )
    # r10d11['delay'] /= uncoded['delay']

    # R10_d0
    # for lmr in lmrs:
    #     lmr.decodingf = complexity.decoding0
    #     lmr.set_wait_for()
    # r10d0 = droplets.simulate(
    #     partial(delay.delay_mean, overhead=1.020148),
    #     lmrs,
    # )
    # r10d0['delay'] /= uncoded['delay']

    # LT
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
        lmr.set_wait_for()
    lt = droplets.simulate(
        partial(delay.delay_mean, overhead=1.3),
        lmrs,
    )
    lt['delay'] /= uncoded['delay']

    # print(np.absolute(1-r10['delay'] / lt['delay']))
    # return

    # make plot
    plt.figure()
    plt.plot(
        r10['nservers'], r10['delay'], r10_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10')
    plt.plot(optimal['nservers'], optimal['delay'], '-v', markevery=0.2, label='R10 sim. (opt.)')
    plt.plot(
        rr['nservers'], rr['delay'], '--', markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10 sim. (rr)')
    # plt.plot(r10d11['nservers'], r10d11['delay'], r10d11_plot_string, markevery=0.2, label='R10d11')
    # plt.plot(rq['nservers'], rq['delay'], rq_plot_string, markevery=0.2, label='RQ')
    plt.plot(
        lt['nservers'], lt['delay'], lt_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='LT')
    plt.plot(bdc['nservers'], bdc['delay'], bdc_plot_string, markevery=0.2,
             label='BDC [6]')
    plt.plot(
        centralized['nservers'], centralized['delay'],
        centralized_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='R10 cent.')
    # plt.plot(clt['nservers'], clt['delay'],
    #          '-', markevery=0.2, label='LT cent. [9]')
    plt.plot(
        classical['nservers'], classical['delay'], classical_plot_string, markevery=0.2,
        markerfacecolor='none', markeredgewidth=1.0, label='Classical [4]')
    plt.plot(bound['nservers'], bound['delay'], bound_plot_string, markevery=0.2, label='Ideal')
    plt.grid(linestyle='--')
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.xlim(0, 600)
    plt.ylim(0.2, 0.7)
    tikz_save(
        './plots/istc18/workload.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth',
    )
    plt.savefig('./plots/istc18/workload.pdf', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def q_plot():
    lmrs = get_parameters_q()
    # [print(lmr) for lmr in lmrs]
    uncoded = droplets.simulate(delay.delay_uncoded, lmrs)
    # analytic = droplets.simulate(delay.delay_mean, lmrs)
    # analytic['delay'] /= uncoded['delay']
    plt.figure()

    # RQ
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.rq_complexity, reloverhead=1.02)
    rq = droplets.simulate(
        partial(delay.delay_mean, overhead=1.02),
        lmrs,
    )
    rq['delay'] /= uncoded['delay']

    # R10
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
    r10 = droplets.simulate(
        partial(delay.delay_mean, overhead=1.020148),
        lmrs,
    )
    r10['delay'] /= uncoded['delay']

    # LT
    for lmr in lmrs:
        lmr.decodingf = partial(complexity.lt_complexity, reloverhead=1.3)
    lt = droplets.simulate(
        partial(delay.delay_mean, overhead=1.3),
        lmrs,
    )
    lt['delay'] /= uncoded['delay']
    lmr0 = lmrs[0]
    print(lmr0)
    lmr0.set_wait_for()
    print(lmr0)

    # plt.plot(analytic['wait_for'], analytic['delay'], label='Analytic')
    plt.plot(r10['wait_for'], r10['delay'], label='R10')
    plt.plot(rq['wait_for'], rq['delay'], label='RQ')
    plt.plot(lt['wait_for'], lt['delay'], label='LT')
    plt.grid()
    plt.legend()
    plt.show()
    return

def plot_server_pdf():
    '''Plot the probability that there are i servers with a delay at most
    t for 1 <= i <= nservers.

    '''
    lmr = lmr1()
    t = 100000000000
    pdf_sim = delay.server_pdf_empiric(t, lmr)
    pdf_ana = delay.server_pdf(t, lmr)
    plt.figure()
    plt.plot(pdf_sim, label='simulated')
    plt.plot(pdf_ana, label='analytic')
    plt.grid()
    plt.legend()
    plt.show()

def error_plot():
    '''Plot the error of the analytic expression.

    '''
    # lmr = get_parameters_workload()[-1]
    lmr = lmr1()
    lmr.decodingf = complexity.decoding0
    lmr.set_wait_for()
    t1 = delay.delay_estimate(10, lmr)
    t2 = delay.delay_estimate(100000, lmr)
    print(t1, t2)
    t = np.linspace(t1, t2, 100, dtype=int)
    simulated = np.fromiter(
        (delay.drops_empiric(i, lmr, n=1000) for i in t),
        dtype=float,
    )
    print(simulated)
    analytic = np.fromiter(
        (delay.drops_estimate(i, lmr) for i in t),
        dtype=float,
    )
    print(analytic)
    se = np.power(simulated-analytic, 2) / analytic
    err = np.absolute(simulated-analytic) / analytic
    plt.plot(t, err, label='error')

    # erra = np.fromiter(
    #     (delay.delay_estimate_error(i, lmr) for i in t),
    #     dtype=float,
    # )
    # plt.plot(t, erra, label='analytic error')
    # plt.plot(t, simulated, label='simulated')
    # plt.plot(t, analytic, label='analytic')
    plt.grid()
    plt.legend()
    plt.show()
    return

    # uncoded = droplets.simulate(delay.delay_uncoded, lmrs)
    for lmr in lmrs:
        # lmr.decodingf = partial(complexity.r10_complexity, reloverhead=1.02)
        lmr.decodingf = complexity.decoding0
        lmr.set_wait_for()

    simulated = droplets.simulate(
        partial(delay.delay_mean_empiric, overhead=1.1, n=1000),
        lmrs,
    )
    # simulated['delay'] /= uncoded['delay']
    analytic = droplets.simulate(
        partial(delay.delay_mean, overhead=1.1),
        lmrs,
    )
    # analytic['delay'] /= uncoded['delay']

def droplets_plot():
    lmr = lmr1()
    t1 = delay.delay_estimate(1, lmr)
    t2 = delay.delay_estimate(1000, lmr)
    t = np.linspace(t1, t2, dtype=int)
    emp = [delay.drops_empiric(x, lmr, n=10000) for x in t]
    avg = [delay.drops_estimate(x, lmr) for x in t]
    lower = [delay.drops_lower(x, lmr) for x in t]
    upper = [delay.drops_upper(x, lmr) for x in t]
    plt.figure()
    plt.plot(t, emp, label='emp.')
    plt.plot(t, avg, label='avg.')
    plt.plot(t, lower, label='lower')
    plt.plot(t, upper, label='upper')
    plt.grid()
    plt.xlim(0, 1e9)
    plt.ylim(0, 1e3)
    plt.xlabel('time')
    plt.ylabel('droplets')
    plt.legend()
    plt.savefig('./plots/istc18/bounds.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def linearity_plot():
    lmr_t = lmr1()
    lmr_c = lmr1()
    tt = delay.delay_estimate(1, lmr_t)
    tc = delay.delay_estimate(1, lmr_c)
    time = list()
    capacity = list()
    for x in range(11):
        time.append(delay.drops_estimate(tt, lmr_t))
        capacity.append(delay.drops_estimate(tc, lmr_c))
        tt *= 2
        lmr_c.nservers *= 2

    time = np.fromiter(time, dtype=float)
    capacity = np.fromiter(capacity, dtype=float)
    print(((time-capacity)/time).mean())
    plt.figure()
    plt.semilogy(time, '-s', markevery=0.2, label='time')
    plt.semilogy(capacity, '--o', markevery=0.25, label='servers')
    plt.xlabel('num. times K and t is doubled')
    plt.ylabel('num. droplets')
    plt.ylim(1, 1e6)
    plt.xlim(0, 10)
    plt.grid()
    plt.legend()
    plt.savefig('./plots/istc18/linearity.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # [print(lmr) for lmr in get_parameters_straggling()]
    # [print(round(lmr.nrows/lmr.droplet_size)) for lmr in get_parameters_straggling()]
    # [print(lmr) for lmr in get_parameters_workload()]
    # lmrs = get_parameters_workload()
    # dt = lmrs[0].asdtype()
    # plot_server_pdf()
    # plot_mean_delay()
    # droplets_plot()
    # bounds_plot()
    # straggling_plot()
    # estimate_plot()
    # q_plot()
    # raptor_plot()
    # error_plot()
    # linearity_plot()
