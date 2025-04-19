import os
import argparse
import json

import numpy as np

import h5py

from scipy.ndimage import binary_dilation

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", type=str, default="data/neurofinder", help="Path to calcium data")
    parser.add_argument("-n", "--name", type=str, default="regions", help="Name of masks file .json")
    parser.add_argument("-s", "--safety", type=int, default=0, help="Safety margin")
    parser.add_argument("-b", "--background", type=int, default=0, help="Background margin")
    opt =parser.parse_args()

    safety_margin = opt.safety
    bg_margin = opt.background

    h_tracks = h5py.File(os.path.join(opt.root_dir, "traces.h5"), "w")

    video_paths = []
    #load video information
    for n in sorted(os.listdir(opt.root_dir)):
        if "test" in n:
            continue

        #load segmentations
        f = open("../../data/neurofinder/{}/data/{}.json".format(n,opt.name), "r")
        regions = json.load(f)
        f.close

        h = h5py.File("../../data/neurofinder/{}/data/{}.h5".format(n,n), "r")
        shape = h["images"].shape

        mask_map = np.zeros(shape[1:])
        for r in regions:
            coords = np.array(r["coordinates"])

            mask_map[coords[:,0], coords[:,1]] = 1


        dset = h_tracks.create_dataset(n, shape=(len(regions), shape[0]))

        for i,r in tqdm(enumerate(regions)):
            coords = np.array(r["coordinates"])

            mins = np.amin(coords, axis=0)
            maxs = np.amax(coords, axis=0)+1
            #add margins to maximum crop size
            mins = mins-safety_margin-bg_margin
            maxs = maxs+safety_margin+bg_margin

            #check if we exceed data size
            mins = np.maximum(mins, np.zeros(2)).astype(int)
            maxs = np.minimum(maxs, shape[1:]).astype(int)

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

            crop = h["images"][:,mins[0]:maxs[0], mins[1]:maxs[1]]

            #get average signal and histograms
            raw_signal = np.sum(crop*np.expand_dims(mask, axis=0), axis=(1,2))/np.sum(mask)

            if bg_mask is not None:
                bg_signal = np.sum(crop*np.expand_dims(bg_mask, axis=0), axis=(1,2))/np.sum(bg_mask)

                #optimize background scaling s so we can subtract it from raw data
                #n = r - s*bg

            dset[i,:] = raw_signal

        h.close()

    h_tracks.close()
