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
from ca_diffusion.datamodule import DataModule
from ca_diffusion.tools.live_histogram import LiveHistogram1D

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
    model.load_state_dict(sd["state_dict"], strict=True)
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    #instantiate dataset we use for training and maybe validation
    print(cfg.data)
    datamodule = DataModule(**cfg.data)
    datamodule.setup()

    loader = datamodule.train_dataloader()

    #iterate over dataset
    latents = []
    num_z = 16 #TODO: do not hardcode!!!
    latent_hist = LiveHistogram1D((-20.0,20.0), num_bins=1000, name="latent_dist")
    latent_hists = [LiveHistogram1D((-20.0,20.0), num_bins=1000, name="latent_dist") for i in range(num_z)] #TODO: infer channel number from config!
    for it,batch in tqdm(enumerate(loader)):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        data = batch[model.data_key]
        z = model.encode(data).sample()

        latent_hist.update(z.cpu().numpy().flatten())
        for i in range(num_z):
            latent_hists[i].update(z[:,i].cpu().numpy().flatten())

        if it<200:
            latents.append(z.cpu())

        if it==1000:
            break

    latents = torch.cat(latents, dim=0)

    mean = torch.mean(latents)
    std = torch.std(latents)

    latent_statistics = {}
    print("Latent Gaussian approximation:", mean.item(), std.item())
    latent_statistics["mean"] = float(mean.item())
    latent_statistics["std"] = float(std.item())
    for i in range(num_z):
        mean = torch.mean(latents[:,i])
        std = torch.std(latents[:,i])
        latent_statistics["mean_{}".format(i)] = float(mean.item())
        latent_statistics["std_{}".format(i)] = float(std.item())
        print("Latent Gaussian approximation {}:".format(i), mean.item(), std.item())

    f = open(os.path.join(out_path, "latent_statistics.json"), "w")
    json.dump(latent_statistics, f)
    f.close()    

    #plot histograms
    plt.figure()
    plt.title("Latent Space Distribution")
    plt.xlabel("Latent Value z")
    plt.ylabel("Probability p(z)")
    plt.yscale("log")
    b, h = latent_hist.data(normalize=True)
    plt.step(b, h, label="Latent")
    plt.savefig(os.path.join(out_path, "hist_latent_space.png"))
    plt.close()

    plt.figure()
    plt.title("Latent Space Distribution")
    plt.xlabel("Latent Value z")
    plt.ylabel("Probability p(z)")
    plt.yscale("log")
    for i in range(num_z):
        b, h = latent_hists[i].data(normalize=True)
        plt.step(b, h, label="Latent {}".format(i), alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(out_path, "hist_latent_space_channels.png"))
    plt.close()


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