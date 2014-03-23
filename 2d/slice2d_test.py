import math
import random
import numpy
from mcmc_tester import mcmc_test_2d

def inv_dist_2d(x, y):
    return math.exp(-(x * x - x * y + y * y) / 2.0)

# 2D Slice sampler
class Slice2D:
    def __init__(self, init_x, init_y, inv_dist, w):
        self.x        = numpy.array([init_x, init_y])
        self.inv_dist = inv_dist
        self.w        = w
        self.t        = 0

    def update(self):
        pp = self.inv_dist(self.x[0], self.x[1])
        u  = random.random() * pp
        minx = numpy.array(self.x)
        maxx = numpy.array(self.x)
        cur = self.t
        opp = (self.t + 1) % 2
        minx[cur] = minx[cur] - self.w / 2.0
        maxx[cur] = maxx[cur] + self.w / 2.0
        while self.inv_dist(minx[0], minx[1]) > u:
            minx[cur] = minx[cur] - self.w
        while self.inv_dist(maxx[0], maxx[1]) > u:
            maxx[cur] = maxx[cur] + self.w

        candx = numpy.zeros(2)
        while True:
            candx[cur] = random.random() * (maxx[cur] - minx[cur]) + minx[cur]
            candx[opp] = self.x[opp]
            if self.inv_dist(candx[0], candx[1]) > u:
                break
            else:
                if candx[cur] < self.x[cur]:
                    minx[cur] = candx[cur]
                else:
                    maxx[cur] = candx[cur]
        self.x = candx
        self.t = (self.t + 1) % 2

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        slc = Slice2D(0.0, 0.0, inv_dist_2d, 0.5)
        mcmc_test_2d(slc, t, 'slice2d', 'Slice sampling (2D)')
