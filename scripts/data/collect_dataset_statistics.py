import sys
sys.path.append("../..")

import argparse

import os

import numpy as np

from tqdm import tqdm
import h5py
import json
import matplotlib.pyplot as plt

from scipy.ndimage import binary_dilation

from ca_diffusion.tools.live_histogram import LiveHistogram1D

def analyze_video(path, name, out_dir, chunk_size=100):
    try:
        h = h5py.File(os.path.join(path, name, "data", name+".h5"), "r")
    except Exception as e:
        return
    
    if not "images" in list(h.keys()):
        h.close()
        return
    
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    
    shape = h["images"].shape
    max_intensity = np.amax(h["images"])

    #load segmentation map
    f = open(os.path.join(path, name, "data", "regions.json"), "r")
    annotations = json.load(f)

    segmap = np.zeros(shape[1:])
    for a in annotations:
        coords = np.array(a["coordinates"])
        segmap[coords[:,0], coords[:,1]] = 1

    #extend segmap by 2 pixels as safety margin
    safety_segmap = binary_dilation(segmap, np.ones((2,2)), iterations=2)

    p_seg = np.sum(segmap)/(shape[1]*shape[2])

    plt.imsave(os.path.join(out_dir, name, "segmap.png"), segmap)
    plt.imsave(os.path.join(out_dir, name, "mean.png"), np.mean(h["images"], axis=0))
    
    intensity_hist = LiveHistogram1D([0, int(max_intensity)], int(max_intensity), name="intensity")
    intensity_hist2 = LiveHistogram1D([0, int(max_intensity)], int(max_intensity), name="intensity")
    intensity_hist3 = LiveHistogram1D([0, int(max_intensity)], int(max_intensity), name="intensity")
    for i in tqdm(range(100)):
        start = i*chunk_size
        end = min((i+1)*chunk_size, shape[0])

        crop = h["images"][start:end].astype(np.float32)
        intensity_hist.update(crop.flatten())

        crop2 = crop*np.expand_dims(1.0-safety_segmap, axis=0)
        crop2 = crop2.flatten()
        crop2 = crop2[np.where(crop2>0)[0]]
        intensity_hist2.update(crop2.flatten())

        crop3 = crop*np.expand_dims(segmap, axis=0)
        crop3 = crop3.flatten()
        crop3 = crop3[np.where(crop3>0)[0]]
        intensity_hist3.update(crop3.flatten())

        #check if this were the last few frames
        if end<(i+1)*chunk_size:
            break

    #find highest probability peak
    bins, val = intensity_hist.data(False)
    ind = np.argmax(val[20:])+20
    print("Max p(I)={:3f} at I={:.3f}".format(bins[ind], val[ind]))

    bins2, val2 = intensity_hist2.data(False)
    bins3, val3 = intensity_hist3.data(False)

    plt.figure(dpi=300)
    plt.title("Max p(I) = {}".format(int(bins[ind]-0.5)))
    plt.xlabel("Intensity I")
    plt.ylabel("Probability p(I)")
    plt.yscale("log")
    plt.step(bins, val)
    plt.step(bins2, val2, alpha=0.5)
    plt.step(bins3, val3, alpha=0.5)
    plt.axvline(bins[ind], color="red")
    plt.savefig(os.path.join(out_dir, name, "intensity_hist.png"))
    plt.close()

    np.savez(os.path.join(out_dir, name, "intensity_hist.npz"), bins-0.5, val)


    h.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, default="data/neurofinder", help="Path to calcium data")
    parser.add_argument("-o", "--output_dir", type=str, default="./statistics", help="Output directory")
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    video_paths = []
    #load video information
    for f in sorted(os.listdir(opt.root_dir)):
        if "test" in f:
            continue
        analyze_video(opt.root_dir, f, opt.output_dir)


    

