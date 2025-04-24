import os
import numpy as np
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn

import h5py
import json

from tqdm import tqdm

import matplotlib.pyplot as plt

if __name__ == "__main__":
    #load video data
    h = h5py.File("../../data/neurofinder/00.00/data/00.00.h5", "r")
    frames = h["images"] #[:1000]
    h.close()

    #load regions
    f = open("../../data/neurofinder/00.00/data/regions.json", "r")
    regions = json.load(f)
    f.close

    mask_map = np.zeros(frames.shape[1:])
    for r in regions:
        coords = np.array(r["coordinates"])

        mask_map[coords[:,0], coords[:,1]] = 1

    safety_margin = 2
    bg_margin = 10
    os.makedirs("traces", exist_ok=True)
    for i,r in tqdm(enumerate(regions)):
        coords = np.array(r["coordinates"])

        mins = np.amin(coords, axis=0)
        maxs = np.amax(coords, axis=0)+1
        #add margins to maximum crop size
        mins = mins-safety_margin-bg_margin
        maxs = maxs+safety_margin+bg_margin

        #check if we exceed data size
        mins = np.maximum(mins, np.zeros(2)).astype(int)
        maxs = np.minimum(maxs, frames.shape[1:]).astype(int)

        mask = np.zeros((maxs-mins).astype(int))
        mask[coords[:,0]-mins[0], coords[:,1]-mins[1]] = 1.0

        #perform opening operations to enlarge masks to get safety and background mask if necessary
        safety_mask = None
        if safety_margin>0 and bg_margin>0: #we only need safety margin for bg
            safety_mask = np.copy(mask_map[mins[0]:maxs[0], mins[1]:maxs[1]]) #calculate safety margin around every mask
            safety_mask = binary_dilation(safety_mask, iterations=safety_margin).astype(float)
    
        bg_mask = None
        if bg_margin>0:
            bg_mask = np.copy(mask)
            bg_mask = binary_dilation(bg_mask, iterations=safety_margin+bg_margin).astype(float)
            if safety_mask is not None:
                bg_mask = bg_mask-safety_mask
            else:
                bg_mask = bg_mask-mask_map[mins[0]:maxs[0], mins[1]:maxs[1]]
            bg_mask[bg_mask<0] = 0.0

        crop = frames[:,mins[0]:maxs[0], mins[1]:maxs[1]]

        #get average signal and histograms
        raw_signal = np.sum(crop*np.expand_dims(mask, axis=0), axis=(1,2))/np.sum(mask)
        raw_signal_dif = None

        r = 1.0
        if bg_mask is not None:
            bg_signal = np.sum(crop*np.expand_dims(bg_mask, axis=0), axis=(1,2))/np.sum(bg_mask)

            #optimize
            raw_tens = torch.FloatTensor(raw_signal)
            bg_tens = torch.FloatTensor(bg_signal)
            scale = nn.Parameter(torch.ones(1))
            optim = torch.optim.Adam([scale], lr=1e-3)

            for it in range(2000):
                optim.zero_grad()
                diff = raw_tens-scale*bg_tens

                loss = torch.mean(diff**2)
                loss.backward()
                optim.step()

            r = scale.item()

        
        neuron_signal = raw_signal-r*bg_signal
        neuron_signal[neuron_signal<0] = 0.0


        plt.figure(dpi=300)
        plt.xlabel("Time")
        plt.ylabel("Intensity")
        plt.title("Traces (r={:.3f})".format(r))
        plt.plot(raw_signal, label="Raw Signal", alpha=0.5)
        plt.plot(bg_signal*r, label="BG Signal", alpha=0.5)
        plt.plot(neuron_signal, label="Neuron Signal", alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join("traces", "trace_{}.png".format(i)))
        plt.close()

            


    

    