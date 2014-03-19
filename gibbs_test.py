from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math

# 2D Gibbs Sampler
class Gibbs:
    def __init__(self, init_x, init_y):
        self.x        = init_x
        self.y        = init_y
        self.t        = 0

    def update(self):
        if(self.t == 0):
            self.x = Gibbs.rand_norm(0.5 * self.y, 1.0)
        else:
            self.y = Gibbs.rand_norm(0.5 * self.x, 1.0)
        self.t = (self.t + 1) % 2

    @staticmethod
    def rand_norm(mu, sigma):
        r1 = random.random()
        r2 = random.random()
        z = math.sqrt(-2.0 * math.log(r1)) * math.sin(2.0 * math.pi * r2)
        return mu + sigma * z

# 2D Histogram
class Hist2D:
    def __init__(self, minx, miny, maxx, maxy, nbins):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.nbins = nbins
        self.spanx = (maxx - minx) / nbins
        self.spany = (maxy - miny) / nbins
        self.bins  = [[0] * nbins for i in range(nbins)]

    def set_value(self, x, y):
        bx = int((x - self.minx) / self.spanx)
        by = int((y - self.miny) / self.spany)
        if bx >=0 and by >= 0 and bx < self.nbins and by < self.nbins:
            self.bins[bx][by] += 1

    def get(self, x, y):
        return self.bins[x][y]

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
    hist = Hist2D(minx, miny, maxx, maxy, nbins)
    for i in range(trial):
        hist.set_value(gibbs.x, gibbs.y)
        gibbs.update()

    xs = [0.0] * nbins * nbins
    ys = [0.0] * nbins * nbins
    zs = [0.0] * nbins * nbins
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for y in range(nbins):
        for x in range(nbins):
            i = y * nbins + x
            xs[i] = x * hist.spanx + hist.minx
            ys[i] = y * hist.spany + hist.miny
            zs[i] = hist.get(x, y) / (trial * hist.spanx * hist.spany)

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
    for t in [1000, 5000, 10000, 50000, 100000, 500000]:
        gibbs_test(0.0, 0.0, t)