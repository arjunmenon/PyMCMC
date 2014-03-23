import math
import time
import hist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mcmc_test_1d(mcmc_inst, trial, name, graph_title):
    # instantiate MCMC object
    burn = int(trial / 10)

    # instantiate histogram object
    minx = -5.0
    maxx =  5.0
    nbins = 40
    histo = hist.Hist(minx, maxx, nbins)

    # burn-in
    for i in range(burn):
        mcmc_inst.update()

    # MCMC simulation
    t1 = time.clock()
    for i in range(trial):
        histo.set_value(mcmc_inst.x)
        mcmc_inst.update()
    t2 = time.clock()
    print('time = %f' % (1000.0 * (t2 - t1) / trial))

    # show histogram
    ys = [histo[i] / (trial * histo.span) for i in range(nbins)]
    xs = [histo.span * i + minx for i in range(nbins)]
    plt.bar(xs, ys, width = histo.span)

    # compute error
    err = 0.0
    for i in range(nbins):
        diff = mcmc_inst.inv_dist(xs[i]) - ys[i]
        err += diff * diff
    print('error = %f' % math.sqrt(err / nbins))

    # show invariant distributin
    ndiv = 400
    span = (maxx - minx) / ndiv
    xs = [span * i + minx for i in range(ndiv)]
    zs = [mcmc_inst.inv_dist(xs[i]) for i in range(ndiv)]
    plt.plot(xs, zs, 'r-')

    plt.suptitle('%s: %d samples' % (graph_title, trial), size='18')
    plt.savefig('%s_%d.png' % (name, trial))
    plt.show()

def mcmc_test_2d(mcmc_inst, trial, name, graph_title):
    burn = int(trial / 10)

    # burn-in
    for i in range(burn):
        mcmc_inst.update()

    nbins = 20
    minx = -3.0
    miny = -3.0
    maxx = 3.0
    maxy = 3.0
    histo = hist.Hist2D(minx, miny, maxx, maxy, nbins)

    # MCMC simulation
    t1 = time.clock()
    for i in range(trial):
        histo.set_value(mcmc_inst.x[0], mcmc_inst.x[1])
        mcmc_inst.update()
    t2 = time.clock()
    print('time = %f' % (1000.0 * (t2 - t1) / trial))

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
    plt.suptitle('%s: %d samples' % (graph_title, trial), size='18')
    plt.savefig('%s_%d.png' % (name, trial))
    plt.show()
