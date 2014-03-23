import math
import random
import numpy
from mcmc_tester import mcmc_test_2d


def inv_dist_2d(x, y):
    return math.exp(-(x * x - x * y + y * y) / 2.0)

# 2D Metropolis Hastings
class MH2D:
    def __init__(self, init_x, init_y, inv_dist, prop_sigma):
        self.x        = numpy.array([init_x, init_y])
        self.inv_dist = inv_dist
        self.sigma    = prop_sigma

    def update(self):
        cand = numpy.random.multivariate_normal(self.x, self.sigma * numpy.eye(2))
        alph = self.inv_dist(cand[0], cand[1]) / (self.inv_dist(self.x[0], self.x[1]) + 1.0e-8)
        alph = min(1.0, alph)
        self.x = cand if random.random() < alph else self.x

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        mh = MH2D(0.0, 0.0, inv_dist_2d, 1.0)
        mcmc_test_2d(mh, t, 'mh_2d', 'M-H method (2D)')