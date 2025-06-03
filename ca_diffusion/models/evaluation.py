import os
from contextlib import contextmanager

import torch
import torch.nn as nn

import lightning.pytorch as pl

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torchmetrics as tm

from ca_diffusion.tools.utils import disabled_train

#TODO: add EMA
class ClassifierEvaluation(pl.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 ignore_param: list=[],
                 ckpt_path: str=None,
                 ignore_ckpt_keys: list=[]
                 ):
        super().__init__()

        self.model = model
        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

        self.accuracy = tm.Accuracy()

        