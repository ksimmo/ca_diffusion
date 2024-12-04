import torch
import torch.nn as nn

import lightning.pytorch as pl

class DiffusionModel(pl.LightningModule):
    def __init__(model: nn.Module,
                 optimizer: torch.optim.Optimizer):
        super().__init__()

        self.model = model

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizer(self):
        pass