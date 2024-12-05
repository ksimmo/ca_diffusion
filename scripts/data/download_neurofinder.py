import os
import io
import argparse

import wget
from zipfile import ZipFile
import h5py
import tifffile
import json
from tqdm import tqdm

import numpy as np

names_train = ["00.00", "00.01", "00.02", "00.03", "00.04", "00.05", "00.06", "00.07", "00.08", "00.09", "00.10", "00.11",
"01.00", "01.01", "02.00", "02.01", "03.00", "04.00", "04.01"]

names_test = ["00.00.test", "00.01.test", "01.00.test", "01.01.test", "02.00.test", "02.01.test", "03.00.test", "04.00.test", "04.01.test"]

url = "https://s3.amazonaws.com/neuro.datasets/challenges/neurofinder/neurofinder."

names = names_train + names_test

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="../../data", help="where to extract neurofinder data")
    opt = parser.parse_args()

    save_path = os.path.join(opt.dir, "neurofinder")

    for n in names:
        print("[+]Downloading {}".format(n))
        wget.download(url+n+".zip", "temp.zip")
        
        #extract .tiff files to h5py
        with ZipFile("temp.zip", "r") as obj:
            #get list of names
            namelist = obj.namelist()
            namelist.sort()

            os.makedirs(os.path.join(save_path, n, "data"), exist_ok=True)
            os.makedirs(os.path.join(save_path, n, "analysis"), exist_ok=True)
            
            
            #extract information file
            infopath = "neurofinder.{}/info.json".format(n)
            with open(os.path.join(save_path, n, "data", "info.json"), 'wb') as f:
                f.write(obj.read(infopath))
            #load information
            f = open(os.path.join(save_path, n, "data", "info.json"), "r")
            infos = json.load(f)
            f.close()

            #extract frames
            tiffs = [x for x in namelist if ".tiff" in x]
            
            numtiffs = len(tiffs)
            extracted = []
            for it,tf in tqdm(enumerate(tiffs)):
                img = tifffile.imread(io.BytesIO(obj.read(tf))).astype(np.float32)
                if it==0:
                    fil = h5py.File(os.path.join(save_path, n, "data", "{}.h5".format(n)), "w")
                    dset = fil.create_dataset("images", shape=(len(tiffs), img.shape[0], img.shape[1]), dtype=np.float32)
                dset[it,:,:] = img
                
            #save dataset attributes
            for k,v in infos.items():
                dset.attrs[k] = v
                
            dset.attrs["mean_intensity"] = np.mean(dset[()])
            dset.attrs["std_intensity"] = np.std(dset[()])
            fil.close()
            
            if not "test" in n:
                regionpath = "neurofinder.{}/regions/regions.json".format(n)
                with open(os.path.join(save_path, n, "data", "regions.json"), 'wb') as f:
                    f.write(obj.read(regionpath))
            
        #finished remove zip file
        os.remove("temp.zip")
