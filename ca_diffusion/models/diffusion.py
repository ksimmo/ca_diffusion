import os

import torch
import torch.nn as nn

import lightning.pytorch as pl

from omegaconf import OmegaConf
from hydra.utils import instantiate

from ca_diffusion.tools.utils import disabled_train
from ca_diffusion.modules.transforms import Transform

#TODO: add EMA
class DiffusionModel(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 diffusor: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 data_key: str="image",
                 ignore_param: list=[]
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "diffusor"]+ignore_param, logger=False) #do not add hyperparams to logger

        self.data_key = data_key

        self.model = model
        self.diffusor = diffusor

    def precompute(self, batch):
        return batch

    def postcompute(self, x):
        return x

    def training_step(self, batch, batch_idx):
        log = {}
        batch = self.precompute(batch)
        loss, loss_dict = self.diffusor(self.model, batch[self.data_key])
        for k, v in loss_dict.items():
            log["train/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    #uncomment to search for unused parameters or inspect gradients
    """
    def on_after_backward(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(name)
            #else:
            #    print(name, torch.mean(param.grad))
    """
    
    def validation_step(self, batch, batch_idx):
        log = {}
        batch = self.precompute(batch)
        loss, loss_dict = self.diffusor(self.model, batch[self.data_key])
        for k, v in loss_dict.items():
            log["val/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = list(self.model.parameters()) + list(self.diffusor.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                             "interval": "step",
                                                             "frequency": 1}}
        return {"optimizer": optimizer}

    
    def log_images(self, batch, **kwargs):
        gt = batch[self.data_key]
        batch = self.precompute(batch)
        noiseshape = batch[self.data_key].size()
        sample = self.diffusor.sample(self.model, self.diffusor.sample_noise(noiseshape).to(self.device), "euler")
        sample = self.postcompute(sample)
        sample2 = self.diffusor.sample(self.model, self.diffusor.sample_noise(noiseshape).to(self.device), "euler")
        sample2 = self.postcompute(sample2)

        log = {}
        log["samples"] = torch.cat([gt.unsqueeze(2), sample.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
        return log



class LatentDiffusionModel(DiffusionModel):
    def __init__(self,
                 first_stage_ckpt: str,
                 latent_transform_args: dict={},
                 precomputed_latents: bool=False,        #does our dataloader already return precomputed latents?
                 **kwargs):
        super().__init__(**kwargs)

        confpath = os.path.join(os.path.dirname(first_stage_ckpt), "..", ".hydra", "config.yaml")
        conf = OmegaConf.load(confpath).model
        conf["ckpt_path"] = first_stage_ckpt

        self.first_stage = instantiate(conf)
        self.first_stage.eval()
        self.first_stage.train = disabled_train
        for param in self.first_stage.parameters():
            param.requires_grad = False

        self.latent_transform = Transform(**latent_transform_args)
        self.precomputed_latents = precomputed_latents

    def precompute(self, batch):
        #precompute all values here and replace the batch with precomputed values
        if not self.precomputed_latents:
            batch[self.data_key] = self.first_stage.encode(batch[self.data_key]).sample()
        batch[self.data_key] = self.latent_transform.forward(batch[self.data_key])
        return batch

    def postcompute(self, x):
        x = self.latent_transform.backward(x)
        return self.first_stage.decode(x)
    
    def on_save_checkpoint(self, checkpoint):
        #TODO: check if this works with deepspeed and FSDP as well!
        #do not save weights from first stage model in ckpt file
        todel = []
        for k in checkpoint["state_dict"].keys():
            if k.startswith("first_stage"):
                todel.append(k)
        for k in todel:    
            del checkpoint["state_dict"][k]

