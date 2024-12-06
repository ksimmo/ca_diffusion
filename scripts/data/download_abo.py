import os
import argparse

import numpy as np

import json
import h5py

import boto3
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, default="../../data", help="where to extract abo data")
    parser.add_argument("-l", "--length", type=int, default=10000, help="maximum length of video, -1 for full length")
    opt = parser.parse_args()


    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-west-2")

    paginator = s3.get_paginator("list_objects_v2")

    def keys(bucket_name, prefix='/', delimiter='/', start_after=''):
        prefix = prefix[1:] if prefix.startswith(delimiter) else prefix
        start_after = (start_after or prefix) if prefix.endswith(delimiter) else start_after
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix, StartAfter=start_after):
            for content in page.get('Contents', ()):
                yield content['Key']
                
    #for k in keys("allen-brain-observatory", "visual-coding-2p"):
    #    print(k)

    experiment_ids = []
    for k in keys("allen-brain-observatory", "visual-coding-2p/ophys_movies"):
        temp = k.rsplit("_",1)[1]
        experiment_ids.append(temp[:-3])


    #first of all download experiment overview
    s3.download_file("allen-brain-observatory","visual-coding-2p/experiment_containers.json","experiment_containers.json")
    s3.download_file("allen-brain-observatory","visual-coding-2p/ophys_experiments.json","ophys_experiments.json")

    save_path = os.path.join(opt.dir, "neurofinder")
    os.makedirs(save_path, exist_ok=True)

    f = open("ophys_experiments.json", "r")
    experiments = json.load(f)
    f.close()
    os.replace("ophys_experiments.json", os.path.join(save_path, "ophys_experiments.json"))

    it = 0
    for e in experiments: #loop over experiments and maybe filter
        name = str(e["id"])
        if name not in experiment_ids:
            continue

        os.makedirs(os.path.join(save_path, name, "data"), exist_ok=True)
        os.makedirs(os.path.join(save_path, name, "analysis"), exist_ok=True)

        #if it>=20:
        #    break

        #filter if needed
        print("Downloading {}:{} ...".format(it, e["id"]))
        
        #download nwb file
        s3.download_file("allen-brain-observatory", "visual-coding-2p/ophys_experiment_data/{}.nwb".format(e["id"]), "{}.nwb".format(e["id"]))
        s3.download_file("allen-brain-observatory", "visual-coding-2p/ophys_movies/ophys_experiment_{}.h5".format(e["id"]), "{}.h5".format(e["id"]))

        fe = h5py.File("{}.nwb".format(e["id"]), "r")
        pixel_size = np.array(fe["general"]["pixel_size"]).astype(str)
        tstruct = np.array(fe["general"]["targeted_structure"]).astype(str)

        #read masks
        rois = [r for r in list(fe["processing"]["brain_observatory_pipeline"]["ImageSegmentation"]["imaging_plane_1"].keys()) if ("roi" in r and not "list" in r)]
        #get each roi
        regions = []
        for r in rois:
            roi = fe["processing"]["brain_observatory_pipeline"]["ImageSegmentation"]["imaging_plane_1"][r]
            pixels = np.array(roi["pix_mask"])
            #convert to neurofinder

            region = dict()
            region["id"] = r
            region["coordinates"] = pixels.tolist()

            regions.append(region)

        #save masks
        #TODO: abo has x and y swapped, change!!!
        temp = open(os.path.join(save_path, name, "data", "regions.json"), "w+")
        json.dump(regions, temp)
        temp.close()

        infos = dict()
        #infos["pixel_size"] = pixel_size
        infos["depth-microns"] = e["imaging_depth"]
        infos["rate-hz"] = 30
        #infos["indicator"] = "GCaMP6"


        #load file
        temp = h5py.File("{}.h5".format(e["id"]), "r")
        length = min(temp["data"].shape[0], opt.length) if opt.length>0 else temp["data"].shape[0]
        frames = np.array(temp["data"][:length,:,:])

        infos["dimensions"] = [frames.shape[1], frames.shape[2], frames.shape[0]]


        #create file
        fil = h5py.File(os.path.join(save_path, name, "data", "{}.h5".format(e["id"])), "w")
        dset = fil.create_dataset("images", shape=frames.shape, dtype=np.float32)
        for i in tqdm(range(frames.shape[0])): #copy over
            dset[i,:,:] = frames[i].astype(np.float32)
            
            
        #save dataset attributes
        for k,v in infos.items():
            dset.attrs[k] = np.array(v)
            
        dset.attrs["mean_intensity"] = np.mean(frames)
        dset.attrs["std_intensity"] = np.std(frames)
        fil.close()

        temp.close()

        #remove files
        os.remove("{}.nwb".format(e["id"]))
        os.remove("{}.h5".format(e["id"]))

        it = it+1
