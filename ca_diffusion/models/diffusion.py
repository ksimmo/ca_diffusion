import torch
import torch.nn as nn

import lightning.pytorch as pl

from ca_diffusion.tools.utils import disabled_train

#TODO: add EMA
class DiffusionModel(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 diffusor: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 data_key: str="image"
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "diffusor"], logger=False) #do net hyperparams to logger

        self.data_key = data_key

        self.model = model
        self.diffusor = diffusor

    def precompute(self, batch):
        return batch

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
        sample = self.diffusor.sample(self.model, self.diffusor.sample_noise(gt.size()).to(self.device), "euler")
        sample2 = self.diffusor.sample(self.model, self.diffusor.sample_noise(gt.size()).to(self.device), "euler")

        log = {}
        log["samples"] = torch.cat([gt.unsqueeze(2), sample.unsqueeze(2), sample2.unsqueeze(2)], dim=2)
        return log



class LatentDiffusionModel(DiffusionModel):
    def __init__(self,
                 first_stage: nn.Module,
                 latent_data: False,        #does our dataloader already return precomputed latents?
                 **kwargs):
        super().__init__(ignore_param=["first_stage"], **kwargs)

        self.first_stage = first_stage
        self.first_stage.eval()
        self.first_stage.train = disabled_train
        for param in self.first_stage.required_parameters():
            param.requires_grad = False

    def precompute(self, batch):
        #precompute all values here and replace the batch with precomputed values
        batch[self.data_key] = self.first_stage.encode(batch[self.data_key]).sample()
        return batch

