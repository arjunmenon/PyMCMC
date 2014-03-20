import math
import random
import mcmc
import matplotlib.pyplot as plt

# invariant distribution
def inv_dist(x):
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

def slice_test(init_x, trial, inv_dist, w):
    slc = Slice(init_x, inv_dist, w)
    burn = int(trial / 10)

    # instantiate histogram object
    minx = -5.0
    maxx =  5.0
    nbins = 40
    hist = mcmc.Hist(minx, maxx, nbins)

    # burn-in
    for i in range(burn):
        slc.update()

    # M-H simulation
    for i in range(trial):
        hist.set_value(slc.x)
        slc.update()

    # show histogram
    ys = [hist[i] / (trial * hist.span) for i in range(nbins)]
    xs = [hist.span * i + minx for i in range(nbins)]
    plt.bar(xs, ys, width = hist.span)

    # show invariant distributin
    ndiv = 400
    span = (maxx - minx) / ndiv
    xs = [span * i + minx for i in range(ndiv)]
    zs = [inv_dist(xs[i]) for i in range(ndiv)]
    plt.plot(xs, zs, 'r-')

    plt.suptitle('Slice sampling: %d samples' % trial, size='18')
    plt.savefig('slice_%d.png' % trial)
    plt.show()

if __name__=='__main__':
    for t in [1000, 5000, 10000, 50000, 100000]:
        slice_test(0.0, t, inv_dist, 0.5)