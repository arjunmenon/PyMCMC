import numpy
from mcmc_tester import mcmc_test_2d

# 2D Gibbs Sampler
class Gibbs:
    def __init__(self, init_x, init_y):
        self.x = numpy.array([init_x, init_y])
        self.t = 0

    def update(self):
        cur = self.t
        opp = (self.t + 1) % 2
        self.x[cur] = numpy.random.randn() * 1.0 + self.x[opp] * 0.5
        self.t = (self.t + 1) % 2

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        gibbs = Gibbs(0.0, 0.0)
        mcmc_test_2d(gibbs, t, 'gibbs', 'Gibbs sampling (2D)')
