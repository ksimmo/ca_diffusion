import torch
import torch.nn as nn

import lightning.pytorch as pl

class DiffusionModel(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 diffusor: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None):
        super().__init__()

        self.save_hyperparameters(ignore=["model", "diffusor"], logger=False) #do net hyperparams to logger

        self.model = model
        self.diffusor = diffusor

    def training_step(self, batch, batch_idx):
        log = {}
        loss, loss_dict = self.diffusor(self.model, batch["data"])
        for k, v in loss_dict.items():
            log["train/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        log = {}
        loss, loss_dict = self.diffusor(self.model, batch["data"])
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