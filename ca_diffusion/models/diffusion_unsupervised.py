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

        self.save_hyperparameters(logger=False) #do net hyperparams to logger

        self.model = model
        self.diffusor = diffusor

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizer(self):
        params = list(self.model.parameters(), self.diffusor.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                             "interval": "step",
                                                             "frequency": 1}}
        return {"optimizer": optimizer}