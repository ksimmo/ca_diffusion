from functools import reduce

import torch
import torch.nn as nn

from ca_diffusion.modules.utils import zero_init, apply_wd, shape_val

#taken from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
def rms_norm(x, scale, eps=1e-6):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)
    
class AdaRMSNorm(nn.Module):
    def __init__(self, channels, context_channels, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.scale = apply_wd(zero_init(nn.Linear(context_channels, channels, bias=False)))

    def forward(self, x, context):
        return rms_norm(x, self.scale(shape_val(context, x, pos=-2))+1.0, self.eps)