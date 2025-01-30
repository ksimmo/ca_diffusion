import sys
sys.path.append("../..")

import os
import argparse
from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
from lightning.pytorch import seed_everything

from hydra.utils import instantiate

import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import imageio
import h5py

from ca_diffusion.tools.plotting_env import initialize_matplotlib

def main(ckpt_path: str, deviceid: int=-1):
    #create evaludation directory from ckpt_path
    run_dir = os.path.join(os.path.dirname(ckpt_path), "..")
    if not os.path.exists(run_dir):
        print("Path to model dir does not exist! {}".format(run_dir))
        exit()
    out_path = os.path.join(run_dir, "eval", "samples")
    os.makedirs(out_path, exist_ok=True)

    cfgpath = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfgpath):
        print("Cannot locate config file {}!".format(cfgpath))
        exit()
    cfg = OmegaConf.load(cfgpath)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility
    #define plotting defaults
    initialize_matplotlib()

    #set up device
    device = torch.device("cuda:{}".format(deviceid) if torch.cuda.is_available() and deviceid>-1 else "cpu")

    #instantiate model
    model = instantiate(OmegaConf.to_container(cfg.model))
    #load from checkpoint
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd["state_dict"], strict=False)
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for it in range(100):
        sample = model.sample(bs=1)
        #sample = sample.clamp(-1.0,1.0)
        sample = sample[0,0].cpu().numpy()
        frames = []
        for i in range(sample.shape[0]):
            fr = sample[i]
            fr = (fr+1.0)*0.5*255.0
            #fr = np.expand_dims(fr, axis=-1)
            frames.append(fr.astype(np.uint8))
        imageio.mimsave(os.path.join(out_path, "sample_{}.gif".format(it)), frames)
    np.save(os.path.join(out_path, "sample_{}.npy".format(it)), sample)

##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    #TODO: make sure to do not save relative paths in the future!
    os.chdir("../..") #go back to root to make sure relative paths work    

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to model checkpoint")
    parser.add_argument("-d", "--deviceid", type=int, default=-1, help="GPU device id, -1 for CPU!")
    opt = parser.parse_args()

    main(opt.path, opt.deviceid)
