import os

import numpy as np

import h5py

import matplotlib.pyplot as plt

if __name__ == "__main__":
    names = []
    for f in sorted(os.listdir("../../outputs")):
        if f!="plots": 
            names.append(f)

    #get length and size of each video
    shapes = []
    bins = []
    vals = []
    for n in names:
        h = h5py.File("../../data/neurofinder/{}/data/{}.h5".format(n,n), "r")
        shapes.append(h["images"].shape)
        h.close()

        obj = np.load("../../outputs/{}/intensity_hist.npz".format(n))
        bins.append(obj["arr_0"])
        vals.append(obj["arr_1"])

    #loaded data
    plt.figure(dpi=300)
    plt.yscale("log")
    for i in range(len(shapes)):
        plt.plot(bins[i]/bins[i][-1], vals[i]/np.prod(shapes[i]), alpha=0.5)
    plt.savefig("../../outputs/plots/intensity_histograms.png")
    plt.close()

    plt.figure(dpi=300)
    plt.yscale("log")
    for i in range(len(shapes)):
        plt.plot(np.log(1.0+bins[i]/bins[i][-1]), vals[i]/np.prod(shapes[i]), alpha=0.5)
    plt.savefig("../../outputs/plots/intensity_histograms_log.png")
    plt.close()

    plt.figure(dpi=300)
    plt.yscale("log")
    for i in range(len(shapes)):
        b = bins[i]/bins[i][-1]*4000
        b = np.floor(b)
        plt.plot(b, vals[i]/np.prod(shapes[i]), alpha=0.5)
    plt.savefig("../../outputs/plots/intensity_histograms_binned.png")
    plt.close()

