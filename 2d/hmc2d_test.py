import math
import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hist

def inv_dist2d(x, y):
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

# M-H test (2D)
def mh2d_test(init_x, init_y, inv_dist, rho, L, trial):
    hmc = HMC2D(init_x, init_y, inv_dist, rho, L)
    burn = int(trial / 10)
    for i in range(burn):
        hmc.update()

    nbins = 20
    minx = -3.0
    miny = -3.0
    maxx = 3.0
    maxy = 3.0
    histo = hist.Hist2D(minx, miny, maxx, maxy, nbins)
    for i in range(trial):
        histo.set_value(hmc.x[0], hmc.x[1])
        hmc.update()

    xs = [0.0] * nbins * nbins
    ys = [0.0] * nbins * nbins
    zs = [0.0] * nbins * nbins
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for y in range(nbins):
        for x in range(nbins):
            i = y * nbins + x
            xs[i] = x * histo.spanx + histo.minx
            ys[i] = y * histo.spany + histo.miny
            zs[i] = histo.get(x, y) / (trial * histo.spanx * histo.spany)

    ax.scatter3D(xs, ys, zs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(minx, maxx)
    ax.set_ylim3d(miny, maxy)
    ax.set_zlim3d(0.0, 0.2)
    plt.suptitle('Hybrid Monte-Carlo (2D): %d samples' % trial, size='18')
    plt.savefig('hmc2d_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        mh2d_test(0.0, 0.0, inv_dist2d, 1.0, 5, t)