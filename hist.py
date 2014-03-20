
# Histogram
class Hist:
    def __init__(self, minx, maxx, nbins):
        self.minx = minx
        self.maxx = maxx
        self.nbins = nbins
        self.span = (maxx - minx) / nbins
        self.bins = [0] * nbins

    def set_value(self, x):
        b = int((x - self.minx) / self.span)
        if b >= 0 and b < self.nbins:
            self.bins[b] += 1

    def __getitem__(self, i):
        return self.bins[i]

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