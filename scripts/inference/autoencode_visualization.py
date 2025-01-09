import sys
sys.path.append("../..")

import os
from omegaconf import DictConfig, OmegaConf

import numpy as np

import torch
from lightning.pytorch import seed_everything

import hydra
from hydra.utils import instantiate

import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import imageio

from ca_diffusion.datamodule import DataModule
from ca_diffusion.tools.plotting_env import initialize_matplotlib

@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility

    #define plotting defaults
    initialize_matplotlib()

    #set up device
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    #create evaludation directory from ckpt_path
    run_dir = os.path.join(os.path.dirname(cfg.ckpt_path), "..")
    if not os.path.exists(run_dir):
        print("Path to model dir does not exist! {}".format(run_dir))
        exit()
    out_path = os.path.join(run_dir, "eval", "samples")
    os.makedirs(out_path, exist_ok=True)

    #instantiate model
    model = instantiate(OmegaConf.to_container(cfg.model))
    #load from checkpoint
    sd = torch.load(cfg.ckpt_path, map_location="cpu")
    print(sd.keys())
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
    real = []
    autoencoded = []
    for it,batch in tqdm(enumerate(loader)):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        data = batch[model.data_key]
        with torch.no_grad():
            rec = model(data)

        real.append(data.cpu())
        autoencoded.append(rec.cpu())

        if it==10:
            break 

    real = torch.cat(real, dim=0)
    autoencoded = torch.cat(autoencoded, dim=0)

    #now save clips
    real_mean = torch.mean(real, dim=2)
    rec_mean = torch.mean(autoencoded, dim=2)
    for i in range(real.size(0)):
        frames = []
        for j in range(real.size(2)):
            fr = torch.cat([real_mean[i,0], real[i,0,j], autoencoded[i,0,j], rec_mean[i,0]], dim=-1) #concatenate along x-axis
            fr = (fr+1.0)*0.5*255.0
            fr = fr.permute(1,2,0)#HxWxC
            frames.append(fr.numpy().astype(np.uint8))
        imageio.mimsave(os.path.join(out_path, "example_{}.gif".format(i)), frames)



##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    #TODO: make sure to do not save relative paths in the future!
    os.chdir("../..") #go back to root to make sure relative paths work    

    main()