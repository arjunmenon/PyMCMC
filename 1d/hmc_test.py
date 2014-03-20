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

class HMC:
    def __init__(self, init_x, inv_dist, rho, L):
        self.x = init_x
        self.inv_dist = inv_dist
        self.rho = rho
        self.L = L

    def update(self):
        u = numpy.random.randn()
        candx = self.x
        candu = u + self.rho * self.delta(candx) * 0.5
        for i in range(self.L):
            candx = candx + self.rho * candu
            fact  = self.rho if i < self.L-1 else self.rho * 0.5
            candu = candu + fact * self.delta(candx)

        ra = self.inv_dist(candx) / self.inv_dist(self.x)
        rb = math.exp(-0.5 * (candu * candu - u * u))
        alph = min(1.0, ra * rb)
        self.x = candx if random.random() < alph else self.x

    def delta(self, x):
        eps = 1.0e-8
        xa = math.log(self.inv_dist(x - eps))
        xb = math.log(self.inv_dist(x + eps))
        return (xb - xa) / (2.0 * eps)

def hmc_test(init_x, trial, inv_dist, rho, L):
    hmc = HMC(init_x, inv_dist, rho, L)
    burn = int(trial / 10)

    # instantiate histogram object
    minx = -5.0
    maxx =  5.0
    nbins = 40
    histo = hist.Hist(minx, maxx, nbins)

    # burn-in
    for i in range(burn):
        hmc.update()

    # M-H simulation
    for i in range(trial):
        histo.set_value(hmc.x)
        hmc.update()

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

    plt.suptitle('Hybrid Monte-Carlo: %d samples' % trial, size='18')
    plt.savefig('hmc_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        hmc_test(0.0, t, inv_dist, 0.5, 5)
