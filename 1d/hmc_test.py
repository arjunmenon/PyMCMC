import math
import random
import numpy
import matplotlib.pyplot as plt

from mcmc_tester import mcmc_test_1d

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

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        hmc = HMC(0.0, inv_dist, 0.5, 5)
        mcmc_test_1d(hmc, t, 'hmc', 'Hybrid Monte Calro')
