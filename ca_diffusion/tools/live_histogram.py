import os
import numpy as np
import matplotlib.pyplot as plt

import pickle

"""
Live Histogram:
Fill a histogram time by time, helpful in cases of large data to avoid RAM overflow
"""

class LiveHistogram1D:
    def __init__(self, range=(0.0, 1.0), num_bins=100, name="histogram"):
        self.name = name
        self.num_bins = num_bins
        self.range = range
        self.hist, edges = np.histogram([], bins=num_bins, range=range)
        self.bins = (edges[:-1] + edges[1:]) / 2.

    def update(self, x):
        hist, edges = np.histogram(x, bins=self.num_bins, range=self.range)
        self.hist += hist

    def data(self, normalize=False):
        data = self.hist
        if normalize:
            data = data/np.sum(data)
        return self.bins, data

    def cumulative(self):
        return self.bins, np.cumsum(self.hist)/np.sum(self.hist)

    def get_percentile_boundary(self, p):
        probabilities = np.cumsum(self.hist)/np.sum(self.hist)
        ind = np.argmin(np.abs(probabilities-p))
        return self.bins[ind]

    def kl(self, hist):
        p = self.hist/np.sum(self.hist)
        q = hist.hist/np.sum(hist.hist)
        return np.sum(p*np.log(p/(q+1e-8)))

    def js(self, hist):
        p = self.hist/np.sum(self.hist)
        q = hist.hist/np.sum(hist.hist)
        m = (p+q)*0.5

        return (np.sum(p*np.log(p/(m+1e-8)))+np.sum(q*np.log(q/(m+1e-8))))*0.5

    def plot(self, normalize="none", norm="linear"):
        data = self.hist
        if normalize:
            data = data/np.sum(data)
        plt.yscale(norm)
        plt.step(data)
        plt.yscale("linear")

    def save(self, path):
        f = open(os.path.join(path, self.name+".pkl"), "wb")
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):  #-> LiveHistogram.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)
    

class LiveHistogram2D:
    def __init__(self, range=(0.0, 1.0), num_bins=100, name="histogram"):
        self.name = name

        if len(range)==2:
            rangex = range
            rangey = range
        else:
            rangex = (range[0], range[1])
            rangey = (range[2], range[3])

        self.num_bins = num_bins
        self.range = (rangex, rangey)
        self.hist, edgesx, edgesy = np.histogram2d([], [], bins=num_bins, range=(rangex, rangey))
        self.binsx = (edgesx[:-1] + edgesx[1:]) / 2.
        self.binsy = (edgesy[:-1] + edgesy[1:]) / 2.

    def update(self, x, y):
        hist, _, _ = np.histogram2d(x, y, bins=self.num_bins, range=self.range)
        self.hist += hist


    def data(self, normalize=False):
        data = self.hist
        if normalize:
            data = data/np.sum(data)

        return self.binsx, self.binsy, data

    def plot(self, normalize="none", norm="linear"):
        #data = (X,Y)
        if normalize=="all":
            data = self.hist/np.sum(self.hist)
        else:
            data = self.hist
        plt.imshow(data.transpose(1,0), origin="lower", extent=[self.range[0][0], self.range[0][1], self.range[1][0], self.range[1][1]], norm=norm)

    def save(self, path):
        f = open(os.path.join(path, self.name+".pkl"), "wb")
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):  #-> LiveHistogram.load(path)
        with open(path, "rb") as f:
            return pickle.load(f)