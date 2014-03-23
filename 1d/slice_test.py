import math
import random
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

# 1D slice sampler
class Slice:
    def __init__(self, init_x, inv_dist, w):
        self.x = init_x
        self.inv_dist = inv_dist
        self.w = w

    def update(self):
        px = self.inv_dist(self.x)
        u  = random.random() * px
        minx = self.x - self.w / 2.0
        maxx = self.x + self.w / 2.0
        while self.inv_dist(minx) > u:
            minx -= self.w
        while self.inv_dist(maxx) > u:
            maxx += self.w

        while True:
            candx = random.random() * (maxx - minx) + minx
            if self.inv_dist(candx) > u:
                break
            else:
                if candx < self.x:
                    minx = candx
                else:
                    maxx = candx
        self.x = candx

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        slc = Slice(0.0, inv_dist_1d, 0.5)
        mcmc_test_1d(slc, t, 'slice', 'Slice sampling')
