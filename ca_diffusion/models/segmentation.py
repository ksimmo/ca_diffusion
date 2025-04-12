import os
from contextlib import contextmanager

import torch
import torch.nn as nn

import lightning.pytorch as pl

from omegaconf import OmegaConf
from hydra.utils import instantiate

from ca_diffusion.tools.utils import disabled_train
from ca_diffusion.modules.ema import EMA

#TODO: add EMA
class BinarySegmentationModel(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 data_key: str="image",
                 target_key: str="segmap",
                 ema_update_steps: int=-1,
                 ema_smoothing_factor: float=0.999,
                 ignore_param: list=[],
                 ckpt_path: str=None,
                 ignore_ckpt_keys: list=[]
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"]+ignore_param, logger=False) #do not add hyperparams to logger

        self.data_key = data_key
        self.target_key = target_key
        self.ema_update_steps = ema_update_steps

        self.model = model

        #set up EMA if necessary
        self.use_ema = self.ema_update_steps>0
        if self.use_ema:
            self.model_ema = EMA(self.model, ema_smoothing_factor)

        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_ckpt_keys)

    def init_from_ckpt(self, ckpt_path, ignore_keys: list=[]):
        """
        Initialize weights only from a checkpoint file 
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        for k in sd.keys():
            for k2 in ignore_keys:
                if k.startswith(k2):
                    print("Removing key {} from state_dict".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing)>0:
            print("Missing keys:", missing)
        if len(unexpected)>0:
            print("Unexpected keys:", unexpected)

    @contextmanager
    def ema(self, desc=None):
        if self.use_ema:
            #save weights for later and use ema weights
            self.model_ema.save(self.model)
            self.model_ema.use_ema(self.model)
            if desc is not None:
                print("{}: Entering EMA!".format(desc))

        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.backup(self.model)
                if desc is not None:
                    print("{}: Leaving EMA!".format(desc))
    def precompute(self, batch):
        return batch

    def postcompute(self, x):
        return x

    def training_step(self, batch, batch_idx):
        log = {}
        batch = self.precompute(batch)
        pred = self.model(batch[self.data_key])
        loss = nn.BCEWithLogitsLoss(pred, batch[self.target_key])
        log["train/bce"] = loss
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
        pred = self.model(batch[self.data_key])
        loss = nn.BCEWithLogitsLoss(pred, batch[self.target_key])
        log["val/bce"] = loss
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def on_before_zero_grad(self, *args, **kwargs):
        if self.use_ema:
            if self.global_step%self.ema_update_steps==0 and self.global_step>1: #update ema every n steps
                self.model_ema(self.model)

    def configure_optimizers(self):
        params = list(self.model.parameters())
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
        segmap = batch[self.target_key]
        segmap = (segmap-0.5)/0.5
        with self.ema("Logging"):
            pred = self.model(batch[self.data_key])
            pred = (torch.sigmoid(pred)-0.5)/0.5

        log = {}
        log["samples"] = torch.cat([gt.unsqueeze(2), segmap.unsqueeze(2), pred.unsqueeze(2)], dim=2)
        return log