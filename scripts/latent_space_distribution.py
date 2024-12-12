from omegaconf import DictConfig, OmegaConf

import torch
from lightning.pytorch import seed_everything

import hydra
from hydra.utils import instantiate

from ca_diffusion.datamodule import DataModule

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility

    #set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #instantiate model
    model = instantiate(OmegaConf.to_container(cfg.model))
    #load from checkpoint
    sd = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(sd["model"]["state_dict"], strict=True)
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    #instantiate dataset we use for training and maybe validation
    datamodule = DataModule(**cfg.data)
    datamodule.setup()

    loader = datamodule.train_dataloader()

    #iterate over dataset
    latents = []
    for it,batch in enumerate(loader):
        for k,v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        data = batch[model.data_key]
        z = model.encode(data).sample()

        latents.append(z.cpu())

        if it==100:
            break

    latents = torch.cat(latents, dim=0)

    mean = torch.mean(latents)
    std = torch.std(latents)
    print("Latent Gaussian approximation:", mean.item(), std.item())



##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    main()