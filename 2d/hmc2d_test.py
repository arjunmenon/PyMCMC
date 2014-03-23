import math
import random
import numpy
from mcmc_tester import mcmc_test_2d

def inv_dist_2d(x, y):
    return math.exp(-(x * x - x * y + y * y) / 2.0)

# 2D Metropolis Hastings
class HMC2D:
    def __init__(self, init_x, init_y, inv_dist, rho, L):
        self.x        = numpy.array([init_x, init_y])
        self.inv_dist = inv_dist
        self.rho      = rho
        self.L        = L

    def update(self):
        u = numpy.random.multivariate_normal(numpy.zeros(2), numpy.eye(2))
        candx = self.x
        candu = u + self.rho * self.delta(candx) * 0.5
        for i in range(self.L):
            candx = candx + self.rho * candu
            fact = self.rho if i < self.L-1 else self.rho * 0.5
            candu = candu + fact * self.delta(candx)
        ra = self.inv_dist(candx[0], candx[1]) / self.inv_dist(self.x[0], self.x[1])
        rb = math.exp(-0.5 * (numpy.dot(candu, candu) - numpy.dot(u, u)))
        alph = min(1.0, ra * rb)
        self.x = candx if random.random() < alph else self.x

    def delta(self, x):
        return numpy.array([-x[0] + 0.5 * x[1], -x[1] + 0.5 * x[0]])

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        hmc = HMC2D(0.0, 0.0, inv_dist_2d, 1.0, 5)
        mcmc_test_2d(hmc, t, 'hmc_2d', 'Hybrid Monte Carlo (2D)')
