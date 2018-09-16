import math
import numpy as np
import matplotlib.pyplot as plt
import pynumeric
import delay
import stats
import droplets
import complexity
import typedefs
import scipy.integrate as integrate

from functools import partial
from numba import jit, njit, prange
from diskcache import FanoutCache

# cache to disk
cache = FanoutCache('./diskcache')

# pyplot setup
plt.style.use('ggplot')
plt.rc('pgf',  texsystem='pdflatex')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['figure.dpi'] = 300

def lmr1():
    '''return a LMR system'''
    return typedefs.lmr_factory(
        straggling_factor=10000,
        nservers=100,
        nrows=10000,
        ncols=100,
        nvectors=100,
        ndroplets=round(10000*1.5),
        droplet_size=100,
        decodingf=complexity.testf,
    )

def lmr2():
    '''return a LMR system'''
    return typedefs.lmr_factory(
        straggling_factor=10000,
        nservers=300,
        nrows=100000,
        ncols=10,
        nvectors=100,
        ndroplets=round(10000*1.5),
        droplet_size=100,
        decodingf=complexity.testf,
    )

success_dtype = np.dtype([
    ('success', np.int64),
    ('samples', np.int64),
])

@cache.memoize(typed=True)
@njit
def success_from_t(t, d, lmr, n_max=math.inf, success_target=100):
    n = 0
    success = 0
    dropletc = lmr['dropletc']
    a = np.zeros(lmr['nservers'])
    while n-success < success_target and n < n_max:
        a = delay.delays(0, lmr, out=a)
        drops = np.floor(np.maximum(t-a, 0)/dropletc).sum()
        if drops >= d: # and a[-1] <= t:
            success += 1
        n += 1
    return success, n

@jit(parallel=True)
def delay_cdf_sim(x, d, lmr):
    '''Evaluate the CDF of the delay at times x.

    Args:

    x: array of delays to evaluate the CDF at.

    d: required number of droplets.

    n: number of samples.

    '''
    cdf = np.zeros(len(x))
    for i in prange(len(x)):
        t = x[i]
        # for i, t in enumerate(x):
        success, samples = success_from_t(t, d, lmr)
        cdf[i] = success/samples
        # print('{}/{}, cdf[i]={}, t={}'.format(i, len(x), cdf[i], t))
        # cdf[i] = result['success']/result['samples']
    return cdf

@jit
def union_bound_1(t, d, lmr):
    '''Lower-bound the probability of computing d droplets within time t.

    Considers several events that all mean the computation will
    succeed. The probability of these events overlap. The bound
    consists of finding the most probable of these events.

    '''
    r = 0
    for g in range(1, lmr['nservers']+1):
        T = t - d * lmr['dropletc'] / g
        if T < 0:
            continue
        v = stats.order_cdf_shiftexp(
            T+lmr['straggling'],
            total=lmr['nservers'],
            order=g,
            parameter=lmr['straggling'],
        )
        r = max(r, v)
    return r

def union_bound_2(t, d, lmr):
    '''Lower-bound the probability of computing d droplets within time t.

    '''
    r = 0
    for g in range(1, lmr['nservers']+1):
        T = t - d * lmr['dropletc'] / g
        if T < 0:
            continue
        v = stats.order_cdf_shiftexp(
            T+lmr['straggling'],
            total=lmr['nservers'],
            order=g,
            parameter=lmr['straggling'],
        )
        print(g, v)
        if g < lmr['nservers']:
            c = stats.order_cdf_shiftexp(
                t-T-lmr['dropletc']+lmr['straggling'],
                total=lmr['nservers']-g,
                order=1,
                parameter=lmr['straggling'],
            )
            v *= c
            print('c', g, c, t, T, t-T)
        print(g, v)
        print()
        r += v
    return r

def bound3_inner(g, t, d, lmr):

    # g servers can always complete the computation by themselves if
    # the g-th server becomes ready at T1 at the latest.
    T1 = t - d * lmr['dropletc'] / g
    if T1 < 0:
        return 0
    v1 = stats.order_cdf_shiftexp(
        T1+lmr['straggling'],
        total=lmr['nservers'],
        order=g,
        parameter=lmr['straggling'],
    )
    r = v1
    if g == lmr['nservers']:
        return r

    f = lambda x: stats.order_pdf_shiftexp(
        x+lmr['straggling'],
        total=lmr['nservers'],
        order=g,
        parameter=lmr['straggling']
    )

    r_inner = 0
    # for g2 in range(g+1, g+2):
    for g2 in range(g+1, lmr['nservers']+1):

        # same as T1 but for g2 servers
        T2 = t - d * lmr['dropletc'] / g2

        # the time at which the g2-th server has to become ready if the
        # g-th server became ready at time t1.
        T3 = lambda t1: g2*t - g*t1 - d*lmr['dropletc']

        F = lambda x: stats.order_cdf_shiftexp(
            T3(x)+lmr['straggling'],
            total=lmr['nservers']-g,
            order=g2-g,
            parameter=lmr['straggling']
        )
        # v1_2, abserr = integrate.quad(lambda x: f(x)*F(x), 0, T1)
        # print(v1/v1_2)
        v2, abserr = integrate.quad(lambda x: f(x)*F(x), T1, T2)
        if not np.isnan(v2):
            r_inner = max(r_inner, v2)

    r += r_inner
    print(v1, r_inner, r)
    return r

def bound3(t, d, lmr):
    r = 0
    for g in range(1, lmr['nservers']+1):
        v = bound3_inner(g, t, d, lmr)
        r = max(r, v)
        # print()
    return min(r, 1.0)

def plot_cdf(num_droplets=2000):

    # x = np.linspace(0, 100*max(straggling_parameter, complexity), 10)
    # time needed to get the droplets
    lmr = lmr2()
    print(lmr)
    t = delay.delay_estimate(num_droplets, lmr)

    r1 = 1
    r2 = 30
    x1 = np.linspace(t/r2, t*r1, 100)
    x2 = np.linspace(t*r1, t*r2, 100)[:83]
    x = np.concatenate((x1, x2))
    print(x)

    # make sure the PDF is correct
    # cdf = np.fromiter((stats.order_cdf_shiftexp(
    #     t+lmr['straggling'],
    #     total=lmr['nservers'],
    #     order=3,
    #     parameter=lmr['straggling']
    # ) for t in x), dtype=float)
    # pdf = np.fromiter((stats.order_pdf_shiftexp(
    #     t+lmr['straggling'],
    #     total=lmr['nservers'],
    #     order=3,
    #     parameter=lmr['straggling']
    # ) for t in x), dtype=float)
    # cdff = lambda x: integrate.quad(
    #     lambda t: stats.order_pdf_shiftexp(
    #         t+lmr['straggling'],
    #         total=lmr['nservers'],
    #         order=3,
    #         parameter=lmr['straggling']
    #     ), 0, x)[0]
    # cdf2 = np.fromiter((cdff(t) for t in x), dtype=float)
    # plt.plot(x, cdf)
    # plt.plot(x, cdf2, '--')
    # plt.grid()
    # plt.show()
    # return

    # simulated
    cdf = delay_cdf_sim(x, num_droplets, lmr)
    # plt.semilogy(x, 1-cdf, label='simulation')
    plt.loglog(x, 1-cdf, label='Simulation')
    print(cdf)

    # bounds
    cdf = np.fromiter((union_bound_1(t, num_droplets, lmr) for t in x), dtype=float)
    # plt.semilogy(x, 1-cdf, 'k--', label='Upper Bound')
    plt.loglog(x, 1-cdf, 'k--', label='Upper Bound')
    print(cdf)

    # print('3', bound3(t, num_droplets, lmr))
    # cdf = np.fromiter((bound3(t, num_droplets, lmr) for t in x), dtype=float)
    # plt.semilogy(x, 1-cdf, label='Bound3')
    # print(cdf)

    # # gamma cdf
    # rv = stats.ShiftexpOrder(
    #     parameter=straggling_parameter,
    #     total=num_servers,
    #     order=num_servers,
    # )
    # t_cdf = rv.mean()-straggling_parameter
    # plt.plot([t_cdf, t_cdf], [0, 1], label='cdf t')

    # pdf = np.diff(cdf)
    # pdf /= pdf.sum()
    # mean_t = integrate.trapz(pdf*x[1:])
    # # mean_t = (pdf*x[1:]).sum()

    # t_ana = delay_mean(num_droplets)
    # plt.plot([t_ana, t_ana], [0, 1], label='analytic')

    # print('empric: {} finv: {} cdf: {} analytic: {}'.format(mean_t, t, t_cdf, t_ana))

    # only order statistics
    # cdf = delay_cdf(x, 1000)
    # plt.plot(x, cdf, label='order statistic')
    plt.xlim((1e8, 1e11))
    plt.ylim((1e-7, 1))
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Pr(\rm{Delay} > t$)')
    plt.tight_layout()
    plt.savefig('./plots/180820/bound.png', dpi='figure', bbox_inches='tight')
    plt.show()
    return

def drops_cdf(straggling_parameter=1, complexity=1):
    '''CDF over the number of droplets computed.

    '''
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

if __name__ == '__main__':
    plot_cdf()
