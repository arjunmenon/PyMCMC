import math
import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hist

def inv_dist2d(x, y):
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

# M-H test (2D)
def mh2d_test(init_x, init_y, inv_dist, prop_sigma, trial):
    mh = MH2D(init_x, init_y, inv_dist, prop_sigma)
    burn = int(trial / 10)
    for i in range(burn):
        mh.update()

    nbins = 20
    minx = -3.0
    miny = -3.0
    maxx = 3.0
    maxy = 3.0
    histo = hist.Hist2D(minx, miny, maxx, maxy, nbins)
    for i in range(trial):
        histo.set_value(mh.x[0], mh.x[1])
        mh.update()

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
    plt.suptitle('M-H method (2D): %d samples' % trial, size='18')
    plt.savefig('mh2d_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        mh2d_test(0.0, 0.0, inv_dist2d, 1.0, t)