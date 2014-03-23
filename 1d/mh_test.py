import math
import random
import numpy
import matplotlib.pyplot as plt

from mcmc_tester import mcmc_test_1d

# invariant distribution
def inv_dist_1d(x):
    r = 0.3
    mu1 = 1.0
    mu2 = -2.0
    x1 = x - mu1
    x2 = x - mu2
    e1 = math.exp(-x1 * x1) / math.sqrt(math.pi)
    e2 = math.exp(-x2 * x2) / math.sqrt(math.pi)
    return r * e1 + (1.0 - r) * e2

# Metropolis Hastings
class MH:
    def __init__(self, init_x, inv_dist, prop_sigma):
        self.x        = init_x
        self.inv_dist = inv_dist
        self.sigma    = prop_sigma

    def update(self):
        candx = numpy.random.randn() * self.sigma + self.x
        curp  = self.inv_dist(self.x)
        candp = self.inv_dist(candx)
        ratio = candp / curp if curp != 0.0 else 1.0
        alph  = min(1.0, ratio)
        r = random.random()
        self.x = candx if r < alph else self.x

if __name__=='__main__':
    sigma = 1.0
    for t in [1000, 5000, 10000, 50000, 100000]:
        mh = MH(0.0, inv_dist_1d, sigma)
        mcmc_test_1d(mh, t, 'mh', 'M-H method')
