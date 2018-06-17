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

from functools import partial
from scipy.stats import expon
from scipy.special import lambertw
from scipy.optimize import minimize
from numba import njit

# pyplot setup
plt.style.use('seaborn-paper')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (3, 3)
plt.rcParams['figure.dpi'] = 300

@njit
def delay_cdf_sim(x, d, n=100):
    '''Evaluate the CDF of the delay at times x.

    Args:

    x: array of delays to evaluate the CDF at.

    d: required number of droplets.

    n: number of samples.

    '''
    cdf = np.zeros(len(x))
    for i, t in enumerate(x):
        for _ in range(n):
            a = get_delay(math.inf)
            drops = np.floor((t-a)/complexity).sum()
            if drops >= d and a[-1] <= t:
                cdf[i] += 1
        cdf[i] /= n
    return cdf

def plot_cdf(num_droplets=2000):
    # x = np.linspace(0, 100*max(straggling_parameter, complexity), 10)
    # time needed to get the droplets
    t = pynumeric.cnuminv(drops_estimate, target=num_droplets)
    plt.plot([t, t], [0, 1], label='avg. t')

    # simulated
    x = np.linspace(t/1.05, t*1.05, 100)
    cdf = delay_cdf_sim(x, num_droplets)
    plt.plot(x, cdf, label='simulation')

    # gamma cdf
    rv = stats.ShiftexpOrder(
        parameter=straggling_parameter,
        total=num_servers,
        order=num_servers,
    )
    t_cdf = rv.mean()-straggling_parameter
    plt.plot([t_cdf, t_cdf], [0, 1], label='cdf t')

    pdf = np.diff(cdf)
    pdf /= pdf.sum()
    mean_t = integrate.trapz(pdf*x[1:])
    # mean_t = (pdf*x[1:]).sum()

    t_ana = delay_mean(num_droplets)
    plt.plot([t_ana, t_ana], [0, 1], label='analytic')

    print('empric: {} finv: {} cdf: {} analytic: {}'.format(mean_t, t, t_cdf, t_ana))

    # only order statistics
    # cdf = delay_cdf(x, 1000)
    # plt.plot(x, cdf, label='order statistic')
    plt.grid()
    plt.legend()
    # plt.show()
    return

def drops_simulation(t, n=100):
    result = 0
    for _ in range(n):
        a = get_delay(t)
        tmp = np.floor((t-a)/complexity)
        if len(tmp) == 0:
            continue
        assert tmp.min() >= 0
        result += tmp.sum()

    return result/n

def time_estimate_K1(d):
    '''Return an approximation of the time t at which d droplets have been
    computed. The inverse of drops_estimate.

    '''
    t = straggling_parameter + complexity/2 + d*complexity
    earg = 2*straggling_parameter
    earg += complexity + 2*d*complexity
    earg /= -2*straggling_parameter
    Warg = math.exp(earg)
    Warg *= -2*straggling_parameter - complexity
    Warg /= 2*straggling_parameter
    t += straggling_parameter * lambertw(Warg)
    return t

def test_estimates(t1=1000):
    d = drops_estimate(t1)
    t2 = time_estimate(d)
    print(t1, t2, d)
    return

def bound4(t):
    bf = lambda x: math.floor((t-x) / complexity)
    f = lambda x: expon.pdf(x, scale=straggling_parameter)*bf(x)
    v, _ = integrate.quad(f, 0, t)
    ub = num_servers * v
    lb = max(ub-num_servers+1, 0)
    return lb, ub

def bound3(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    for i in range(num_servers):
        norm = expon.cdf(t, scale=straggling_parameter)
        v = 1/straggling_parameter*t
        v += math.exp(-1/straggling_parameter*t) - 1
        v *= straggling_parameter
        v /= norm * complexity
        ub = v*(i+1)
        # lb = max(ub-i+1, 0)
        lb = ub-i+1
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb
    return lbt, ubt

def bound2(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    bf = lambda x: (t-x) / complexity
    for i in range(num_servers):
        norm = expon.cdf(t, scale=straggling_parameter)
        f = lambda x: expon.pdf(x, scale=straggling_parameter)/norm*bf(x)
        if norm < np.finfo(float).tiny:
            continue
        v, _ = integrate.quad(f, 0, t)
        # ub = math.floor(v)*(i+1)
        ub = v*(i+1)
        lb = max(ub-i+1, 0)
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb
    return lbt, ubt

def bound(t):
    '''Return a lower and upper bound on the average number of droplets
    computed at time t.

    '''
    pdf = server_pdf(t)
    ubt = 0
    lbt = 0
    # print()
    for i in range(num_servers):
        rv = stats.ExpSum(
            scale=straggling_parameter,
            order=i+1,
        )
        norm = rv.cdf((i+1)*t)
        # s = integrate.quad(lambda x: rv.pdf(x)/norm, 0, (i+1)*t)
        # print('sum={}, norm={}'.format(s, norm))
        if norm < np.finfo(float).tiny:
            continue

        # bf = lambda x: ((i+1)*t-x)/complexity
        # p, _ = integrate.quad(rv.pdf, 0, straggling_parameter)
        # print(p, )
        # ub, _ = integrate.quad(
        #     lambda x: rv.pdf(x)/norm * bf(x),
        #     0,
        #     (i+1)*t,
        # )
        ub = 0
        if t > straggling_parameter:
            ub += (i+1)*(t - straggling_parameter)/complexity

        lb = max(ub-i+1, 0)
        ubt += pdf[i] * ub
        lbt += pdf[i] * lb

    return lbt, ubt

def drops_cdf():
    t = np.linspace(1, 10*max(straggling_parameter, complexity), 100)
    y = [computed_drops(i) for i in t]
    plt.plot(t, [i[0] for i in y], '-o', label='simulation', markevery=0.1)
    # plt.plot(t, [i[1] for i in y], label='lb')
    # plt.plot(t, [i[2] for i in y], label='up')
    # plt.plot(t, [(i[1]+i[2])/2 for i in y], label='(lb+up)/2')

    y = [bound4(i) for i in t]
    # plt.plot(t, [i[0] for i in y], label='analytic lb')
    plt.plot(t, [i[1] for i in y], '-', label='integral')
    # plt.plot(t, [(i[0]+i[1])/2 for i in y], label='analytic mean')

    y = [bound5(i) for i in t]
    plt.plot(t, [i[1] for i in y], '--', label='approximation')
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('avg. number of droplets computed')
    plt.legend()
    plt.show()
    return

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
        decodingf=complexity.decodingf,
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
        decodingf=complexity.decodingf,
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
    # nvectors = round(nrows/ratio / nservers) * nservers
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
        decodingf=complexity.decodingf,
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
        # lmr = find_parameters(
        #     nservers=nservers,
        #     C=C_target,
        #     straggling_factor=i,
        # )
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
        cache='bdc_straggling',
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
    plt.plot(r10['straggling_factor'], r10['delay'], r10_plot_string, markevery=0.2, label='R10')
    # plt.plot(r10d11['straggling_factor'], r10d11['delay'], r10d11_plot_string, markevery=0.2, label='R10d11')
    plt.plot(rq['straggling_factor'], rq['delay'], rq_plot_string, markevery=0.2, label='RQ')
    plt.plot(lt['straggling_factor'], lt['delay'], lt_plot_string, markevery=0.2, label='LT')
    plt.plot(bdc['straggling_factor'], bdc['delay'], bdc_plot_string, markevery=0.2, label='BDC [6]')
    plt.plot(centralized['straggling_factor'], centralized['delay'],
             centralized_plot_string, markevery=0.2, label='R10 cent.')
    # plt.plot(clt['straggling_factor'], clt['delay'],
    #          '-', markevery=0.2, label='LT cent. [9]')
    plt.plot(classical['straggling_factor'], classical['delay'],
             classical_plot_string, markevery=0.2, label='Classical [4]')
    plt.plot(bound['straggling_factor'], bound['delay'], bound_plot_string, markevery=0.2, label='Ideal')

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
    plt.ylim(0, 2)
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
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

r10_plot_string = 'b-o'
r10d11_plot_string = 'g-v'
rq_plot_string = 'g-^'
lt_plot_string = 'm-d'
bdc_plot_string = 'c-*'
centralized_plot_string = 'r-s'
bound_plot_string = 'k-'
classical_plot_string = 'g:'

def optimize_q_bdc(lmr, num_partitions):
    '''Optimize the value of q for the BDC scheme

    '''
    def f(q):
        nonlocal lmr
        q = q[0]
        # q = int(round(q[0]))

        # enforce bounds
        if q > lmr.nservers:
            return q * 1e32
        if q < 1:
            return -(q-2) * 1e32

        decoding = complexity.bdc_decoding_complexity(
            lmr,
            partitions=num_partitions,
        )
        decoding *= lmr.nvectors / q

        # interpolate between the floor and ceil of q
        s1 = stats.ShiftexpOrder(
            parameter=lmr.straggling,
            total=lmr.nservers,
            order=int(math.floor(q)),
        ).mean()
        s2 = stats.ShiftexpOrder(
            parameter=lmr.straggling,
            total=lmr.nservers,
            order=int(math.ceil(q)),
        ).mean()
        straggling = s1 * (math.ceil(q)-q) + s2 * (q-math.floor(q))
        return decoding + straggling

    result = minimize(
        f,
        x0=lmr.nservers-1,
        # bounds=[(1, lmr.nservers)],
        method='Powell',
    )
    wait_for = int(result.x.round())
    delay = result.fun

    # add the overhead due to partitioning
    # this is an upper bound
    delay += lmr.dropletc * num_partitions / 2

    return wait_for, delay

def optimize_bdc(lmr):
    '''Optimize the number of partitions and the number of servers to wait
    for.

    '''
    min_q = None
    min_d = math.inf
    min_T = None
    max_T = int(round(lmr.nrows / lmr.droplet_size))
    for T in range(int(round(max_T*0.9)), max_T+1):
        # if lmr.ndroplets % T != 0:
        #     continue
        q, d = optimize_q_bdc(lmr, T)
        if d < min_d:
            min_d = d
            min_q = q
            min_T = T

    # print('T={}, q={} is optimal for lmr {}'.format(min_T, min_q, lmr))
    return min_q, min_d

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
        cache='bdc',
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
        cache='heuristic',
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
    plt.plot(r10['nservers'], r10['delay'], r10_plot_string, markevery=0.2, label='R10')
    plt.plot(optimal['nservers'], optimal['delay'], '--', markevery=0.2, label='R10 sim. (opt.)')
    plt.plot(rr['nservers'], rr['delay'], '--', markevery=0.2, label='R10 sim. (rr)')
    # plt.plot(r10d11['nservers'], r10d11['delay'], r10d11_plot_string, markevery=0.2, label='R10d11')
    # plt.plot(rq['nservers'], rq['delay'], rq_plot_string, markevery=0.2, label='RQ')
    plt.plot(lt['nservers'], lt['delay'], lt_plot_string, markevery=0.2, label='LT')
    plt.plot(bdc['nservers'], bdc['delay'], bdc_plot_string, markevery=0.2, label='BDC [6]')
    plt.plot(centralized['nservers'], centralized['delay'],
             centralized_plot_string, markevery=0.2, label='R10 cent.')
    plt.plot(clt['nservers'], clt['delay'],
             '-', markevery=0.2, label='LT cent. [9]')
    plt.plot(classical['nservers'], classical['delay'], classical_plot_string, markevery=0.2, label='Classical [4]')
    plt.plot(bound['nservers'], bound['delay'], bound_plot_string, markevery=0.2, label='Ideal')
    plt.grid(linestyle='--')
    plt.legend(framealpha=1, labelspacing=0.1, columnspacing=0.1, ncol=1, loc='best')
    plt.ylabel('$D$')
    plt.xlabel('$K$')
    plt.xlim(0, 600)
    plt.ylim(0.2, 0.7)
    plt.savefig('./plots/istc18/workload_raptor.pdf', dpi='figure', bbox_inches='tight')
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




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # [print(lmr) for lmr in get_parameters_straggling()]
    # [print(round(lmr.nrows/lmr.droplet_size)) for lmr in get_parameters_workload()]
    # [print(lmr) for lmr in get_parameters_workload()]
    # lmrs = get_parameters_workload()
    # dt = lmrs[0].asdtype()
    # plot_server_pdf()
    # plot_mean_delay()
    # straggling_plot()
    # estimate_plot()
    # q_plot()
    raptor_plot()
    # error_plot()
