import math
import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import hist

# 2D Gibbs Sampler
class Gibbs:
    def __init__(self, init_x, init_y):
        self.x        = init_x
        self.y        = init_y
        self.t        = 0

    def update(self):
        if(self.t == 0):
            self.x = numpy.random.randn() * 1.0 + self.y * 0.5
        else:
            self.y = numpy.random.randn() * 1.0 + self.x * 0.5
        self.t = (self.t + 1) % 2

# Gibbs sampler test
def gibbs_test(init_x, init_y, trial):
    gibbs = Gibbs(init_x, init_y)
    burn = int(trial / 10)
    for i in range(burn):
        gibbs.update()

    nbins = 20
    minx = -3.0
    miny = -3.0
    maxx = 3.0
    maxy = 3.0
    histo = hist.Hist2D(minx, miny, maxx, maxy, nbins)
    for i in range(trial):
        histo.set_value(gibbs.x, gibbs.y)
        gibbs.update()

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
    plt.suptitle('Gibbs sampler: %d samples' % trial, size='18')
    plt.savefig('gibbs_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        gibbs_test(0.0, 0.0, t)