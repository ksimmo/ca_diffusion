import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ca_diffusion.modules.utils import zero_init, checkpoint, shape_val
from ca_diffusion.modules.transformer.blocks import AttentionBlock

#currently down and upsampling only support box filter
class Downsample2D(nn.Module):
    def __init__(self, f=[1,1]):
        super().__init__()

        self.pad = (len(f) - 1) // 2

        f = np.array(f)
        f = f/f.sum()
        weights = torch.FloatTensor(np.outer(f,f))
        self.register_buffer("weights", weights.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        return F.conv2d(x, self.weights.repeat(x.size(1),1,1,1), groups=x.size(1), stride=2, padding=(self.pad,))

class Upsample2D(nn.Module):
    def __init__(self, f=[1,1]):
        super().__init__()

        self.pad = (len(f) - 1) // 2

        f = np.array(f)
        f = f/f.sum()
        weights = torch.FloatTensor(np.outer(f,f)*4.0)
        self.register_buffer("weights", weights.unsqueeze(0))

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights.repeat(x.size(1),1,1,1), groups=x.size(1), stride=2, padding=(self.pad,))
    

class ResnetBlock2D(nn.Module):
    def __init__(self, channels_in, channels_out, mode="default", num_groups=32, dropout=0.0, channels_emb=None, scale_shift_norm=True, use_checkpoint=False, compile=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.scale_shift_norm = scale_shift_norm
        self.mode = mode

        if mode not in ["default", "up", "down"]:
            raise NotImplementedError("ResnetBlock mode {} is not supported!".format(mode))
        
        blocks_in = [nn.GroupNorm(num_groups, channels_in), nn.SiLU()]
        if mode=="up":
            blocks_in += [Upsample2D()]
        elif mode=="down":
            blocks_in += [Downsample2D()]
        blocks_in += [nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True)]
        if self.scale_shift_norm:
            blocks_in += [nn.GroupNorm(num_groups, channels_out)]
        self.blocks_in = nn.Sequential(*blocks_in)

        self.emb_layers = None
        if channels_emb is not None:
            self.emb_layers = nn.Linear(channels_emb, channels_out*2 if scale_shift_norm else channels_out, bias=True)

        blocks_out = []
        if not self.scale_shift_norm:
            blocks_out += [nn.GroupNorm(num_groups, channels_out)]
        blocks_out += [nn.SiLU(), nn.Dropout(dropout),
                      zero_init(nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True))]
        self.blocks_out = nn.Sequential(*blocks_out)

        skip = []
        if mode=="up":
            skip += [Upsample2D()]
        elif mode=="down":
            skip += [Downsample2D()]
        if mode!="default" or channels_in!=channels_out:
            skip += [nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1, padding=0, bias=True)]
        
        self.skip = nn.Sequential(*skip) if len(skip)>0 else nn.Identity()

        if compile:
            self._forward = torch.compile(self._forward, fullgraph=True)

    def forward(self, x, emb=None):
        return checkpoint(self._forward, self.use_checkpoint, x, emb)

    def _forward(self, x, emb=None):
        h = self.blocks_in(x)

        if emb is not None and self.emb_layers is not None:
            emb = self.emb_layers(emb)
            emb = shape_val(emb, h, -1)
            if self.scale_shift_norm:
                scale, shift = emb.chunk(2, dim=1)
                h = h*(1.0+scale)+shift
            else:
                h = h+emb

        h = self.blocks_out(h)
        return self.skip(x)+h 
    

class Attention2D(nn.Module):
    def __init__(self, channels, num_heads, head_channels=None, qkv_bias=True, qk_norm=True, dropout=0.0, compile=False, use_checkpoint=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attention = AttentionBlock(channels, num_heads, head_channels=head_channels, qkv_bias=qkv_bias, qk_norm=qk_norm, dropout=dropout, compile=compile)

    def forward(self, x):
        xsize = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        x = checkpoint(self.attention, self.use_checkpoint, x)
        x = rearrange(x, "b (h w) c -> b c h w", h=xsize[-2], w=xsize[-1])
        return x