import numpy as np

import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class PositionEmbedding1D(nn.Module):
    def __init__(self, channels_freq, channels, max_period=10000, repeat_only=False):
        super().__init__()

        self.channels = channels_freq

        self.repeat_only = repeat_only
        if not self.repeat_only:
            self.register_buffer("frequencies", torch.exp(-np.log(max_period) * torch.arange(start=0, end=channels_freq//2) / (channels_freq//2)))

        self.embedder = nn.Sequential(nn.Linear(channels_freq, channels), nn.SiLU(), nn.Linear(channels,channels))

    def forward(self, x):
        if self.repeat_only:
            x = x.unsqueeze(-1).repeat(1,self.channels)
            return self.embedder(x)
        else:
            freqs = self.frequencies.to(torch.float32)
            freqs = shape_val(self.frequencies.to(torch.float32), x, pos=0)
            emb = x.to(torch.float32).unsqueeze(-1)*freqs
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
            return self.embedder(emb.to(x.dtype))
        

#RoPE (mainly taken from k-diffusion: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py)
def _apply_rotary_emb_inplace(x, theta, conj):
    dtype = torch.float32
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    x1_, x2_, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1_ * cos - x2_ * sin
    y2 = x2_ * cos + x1_ * sin
    x1.copy_(y1)
    x2.copy_(y2)

class ApplyRotaryEmbeddingInplace(torch.autograd.Function):
    @staticmethod
    def forward(x, theta, conj):
        _apply_rotary_emb_inplace(x, theta, conj=conj)
        return x

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta, conj = inputs
        ctx.save_for_backward(theta)
        ctx.conj = conj

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        _apply_rotary_emb_inplace(grad_output, theta, conj=not ctx.conj)
        return grad_output, None, None


def apply_rotary_emb_(x, theta):
    return ApplyRotaryEmbeddingInplace.apply(x, theta, False)