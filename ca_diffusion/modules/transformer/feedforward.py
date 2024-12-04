import torch
import torch.nn as nn

from ca_diffusion.modules.utils import zero_init, apply_wd
from ca_diffusion.modules.act import GEGLU, SwiGLU

class FeedForward(nn.Module):
    def __init__(self, channels, expand=4, dropout=0.0, bias=True, act="default"):
        super().__init__()

        inner_channels = channels*int(expand)

        if act=="default":
            self.proj_in = nn.Sequential(apply_wd(nn.Linear(channels, inner_channels)), nn.GELU())
        elif act=="swiglu":
            self.proj_in = apply_wd(SwiGLU(channels, inner_channels))
        elif act=="geglu":
            self.proj_in = apply_wd(GEGLU(channels, inner_channels))
        else:
            raise NotImplementedError("FeedForward layer does not support {} activation".format(act))
        self.dropout = nn.Dropout(dropout)
        self.proj_out = apply_wd(zero_init(nn.Linear(inner_channels, channels, bias=bias)))

    def forward(self, x):
        x = self.proj_in(x)
        x = self.dropout(x)
        return self.proj_out(x)