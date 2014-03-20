import math
import random
import numpy
import matplotlib.pyplot as plt

import hist

# invariant distribution
def inv_dist(x):
    r = 0.3
    mu1 = 1.0
    mu2 = -2.0
    x1 = x - mu1
    x2 = x - mu2
    e1 = math.exp(-x1 * x1) / math.sqrt(math.pi)
    e2 = math.exp(-x2 * x2) / math.sqrt(math.pi)
    return r * e1 + (1.0 - r) * e2

# Simulated anealing
class SA:
    def __init__(self, init_x, inv_dist, prop_sigma, T0, rho):
        self.x        = init_x
        self.inv_dist = inv_dist
        self.sigma    = prop_sigma
        self.T        = T0
        self.rho      = rho

    def update(self):
        candx = numpy.random.randn() * self.sigma + self.x
        curp  = self.inv_dist(self.x) ** (1.0 / self.T)
        candp = self.inv_dist(candx) ** (1.0 / self.T)
        ratio = candp / curp if curp != 0.0 else 1.0
        alph = min(1.0, ratio)
        r = random.random()
        self.x = candx if r < alph else self.x
        self.T = self.T * self.rho

def sa_test(init_x, trial, inv_dist):
    # instantiate M-H object
    burn = int(trial / 10)
    sa = SA(init_x, inv_dist, 1.0, 1.0, 0.99998)

    # instantiate histogram object
    minx = -5.0
    maxx =  5.0
    nbins = 40
    histo = hist.Hist(minx, maxx, nbins)

    # burn-in
    for i in range(burn):
        sa.update()

    # M-H simulation
    for i in range(trial):
        histo.set_value(sa.x)
        sa.update()

    # show histogram
    ys = [histo[i] / (trial * histo.span) for i in range(nbins)]
    xs = [histo.span * i + minx for i in range(nbins)]
    plt.bar(xs, ys, width = histo.span)

    # show invariant distributin
    ndiv = 400
    span = (maxx - minx) / ndiv
    xs = [span * i + minx for i in range(ndiv)]
    zs = [inv_dist(xs[i]) for i in range(ndiv)]
    plt.plot(xs, zs, 'r-')

    plt.suptitle('Simulated annealing: %d samples' % trial, size='18')
    plt.savefig('sa_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    sa_test(0.0, 1000, inv_dist)
