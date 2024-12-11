import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ca_diffusion.modules.utils import zero_init, checkpoint, shape_val
from ca_diffusion.modules.transformer.blocks import AttentionBlock

#wrap conv3d so we can do causal convolution if we need to
class Conv3D(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, causal=False, **kwargs):
        self.causal = causal
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size, kernel_size]
        if not isinstance(stride, (list, tuple)):
            stride = [stride, stride, stride]
        if not isinstance(padding, (list, tuple)):
            padding = [padding, padding, padding]
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation, dilation, dilation]
        pad = padding
        if causal:
            pad[0] = 0 #only disable temporal padding
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, **kwargs)

        self.pad_temporal = 0
        if self.causal and kernel_size[0]>1: #only pad when necessary
            self.pad_temporal = kernel_size[0]-1


    def forward(self, x):
        if self.pad_temporal>0 and self.causal:
            x = F.pad(x, (0,0,0,0,self.pad_temporal,0), mode="replicate")
        return super().forward(x)


class Downsample3D(nn.Module):
    def __init__(self, channels=None, spatial=True, temporal=True, causal=False, learnable=False):
        super().__init__()

        self.learnable = learnable
        self.spatial = spatial
        self.temporal = temporal
        self.causal = causal
        if causal:
            if learnable and spatial:
                self.pool_sp = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)
            elif spatial:
                self.pool_sp = nn.AvgPool2d(2,2)
            else:
                self.pool_sp = nn.Identity()

            if self.temporal:
                self.pool_t = nn.AvgPool3d((2,1,1),(2,1,1))
            else:
                self.pool_t = nn.Identity()
        else:
            if learnable:
                if spatial and temporal:
                    self.pad = (0,1,0,1,0,1)
                    self.pool = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=0)
                elif spatial:
                    self.pad = (0,1,0,1,0,0)
                    self.pool = nn.Conv3d(channels, channels, kernel_size=(1,3,3), stride=(1,2,2), padding=0)
                elif temporal:
                    self.pad = (0,0,0,0,0,1)
                    self.pool = nn.Conv3d(channels, channels, kernel_size=(3,1,1), stride=(2,1,1), padding=0)
            else:
                if spatial and temporal:
                    self.pool = nn.AvgPool3d(2, 2)
                elif spatial:
                    self.pool = nn.AvgPool3d((1,2,2), (1,2,2))
                elif temporal:
                    self.pool = nn.AvgPool3d((2,1,1), (2,1,1))


    def forward(self, x):
        if self.causal:
            B,T = x.size(0), x.size(2)
            if self.temporal and T>1:
                #split
                x_first = x[:,:,0:1]
                x_rest = x[:,:,1:]
                x_rest = self.pool_t(x_rest)
                x = torch.cat([x_first, x_rest], dim=2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            if self.learnable and self.spatial:
                x = F.pad(x, (0,1,0,1), mode="constant", value=0)
            x = self.pool_sp(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        else:
            if self.learnable:
                x = F.pad(x, self.pad, mode="constant", value=0)
                x = self.pool(x)
            else:
                x = self.pool(x)        
        return x

class Upsample3D(nn.Module):
    def __init__(self, channels=None, spatial=True, temporal=True, causal=False, learnable=False):
        super().__init__()

        self.learnable = learnable
        self.spatial = spatial
        self.temporal = temporal
        self.causal = causal

        if causal:
            if learnable and spatial:
                self.unpool_sp = nn.Sequential(nn.Upsample(scale_factor=2.0), nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
            elif spatial:
                self.unpool_sp = nn.Upsample(scale_factor=2.0, mode="nearest")
            else:
                self.unpool_sp = nn.Identity()

            if self.temporal:
                self.unpool_t = nn.Upsample(scale_factor=(2.0,1.0,1.0), mode="nearest")
            else:
                self.unpool_t = nn.Identity()
        else:
            unpool = []
            if spatial and temporal:
                unpool += [nn.Upsample(scale_factor=2.0)]
                if learnable:
                    unpool += [nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1)]
            elif spatial:
                unpool += [nn.Upsample(scale_factor=(1.0, 2.0, 2.0))]
                if learnable:
                    unpool += [nn.Conv3d(channels, channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1))]
            elif temporal:
                unpool += [nn.Upsample(scale_factor=(2.0, 1.0, 1.0))]
                if learnable:
                    unpool += [nn.Conv3d(channels, channels, kernel_size=(3,1,1), stride=1, padding=(1,0,0))]
            self.unpool = nn.Sequential(*unpool)

    def forward(self, x):
        if self.causal:
            B,T = x.size(0), x.size(2)
            if self.temporal and T>1:
                #split
                x_first = x[:,:,0:1]
                x_rest = x[:,:,1:]
                x_rest = self.unpool_t(x_rest)
                x = torch.cat([x_first, x_rest], dim=2)
            #do spatial upsampling if necessary
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.unpool_sp(x)
            x = rearrange(x, "(b t ) c h w -> b c t h w", b=B)
        else:
            x = self.unpool(x)     
        return x

    

class ResnetBlock3D(nn.Module):
    def __init__(self, channels_in, channels_out, mode="default", num_groups=32, dropout=0.0, channels_emb=None, scale_shift_norm=True, use_checkpoint=False, compile=False, 
                 learnable_pool=False, causal=False):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.scale_shift_norm = scale_shift_norm
        self.mode = mode

        if mode not in ["default", "up", "down", "up_sp", "down_sp", "up_tp", "down_temp"]:
            raise NotImplementedError("ResnetBlock mode {} is not supported!".format(mode))
        
        blocks_in = [nn.GroupNorm(num_groups, channels_in), nn.SiLU()]
        if "up" in mode:
            blocks_in += [Upsample3D(channels_in, spatial=False if "tp" in mode else True, temporal=False if "sp" in mode else True, 
                                     causal=causal, learnable=learnable_pool)]
        elif "down" in mode:
            blocks_in += [Downsample3D(channels_in, spatial=False if "tp" in mode else True, temporal=False if "sp" in mode else True, 
                                       causal=causal, learnable=learnable_pool)]
        blocks_in += [Conv3D(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=True, causal=causal)]
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
                      zero_init(Conv3D(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=True, causal=causal))]
        self.blocks_out = nn.Sequential(*blocks_out)

        skip = []
        if "up" in mode:
            skip += [Upsample3D(channels_in, spatial=False if "tp" in mode else True, temporal=False if "sp" in mode else True, 
                                     causal=causal, learnable=False)]
        elif "down" in mode:
            skip += [Downsample3D(channels_in, spatial=False if "tp" in mode else True, temporal=False if "sp" in mode else True, 
                                       causal=causal, learnable=False)]
        if mode!="default" or channels_in!=channels_out:
            skip += [Conv3D(channels_in, channels_out, kernel_size=1, stride=1, padding=0, bias=True, causal=causal)]
        
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
    

class Attention3D(nn.Module):
    def __init__(self, channels, num_heads, head_channels=None, qkv_bias=True, qk_norm=True, dropout=0.0, compile=False, use_checkpoint=False, attn_type="vanilla"):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.attn_type = attn_type

        if attn_type=="vanilla" or attn_type=="spatial" or attn_type=="temporal":
            self.attention = AttentionBlock(channels, num_heads, head_channels=head_channels, qkv_bias=qkv_bias, qk_norm=qk_norm, dropout=dropout, compile=compile)
        elif attn_type=="factorized":
            self.attention_sp = AttentionBlock(channels, num_heads, head_channels=head_channels, qkv_bias=qkv_bias, qk_norm=qk_norm, dropout=dropout, compile=compile)
            self.attention_t = AttentionBlock(channels, num_heads, head_channels=head_channels, qkv_bias=qkv_bias, qk_norm=qk_norm, dropout=dropout, compile=compile)
        else:
            raise NotImplementedError("3D Attention type {} is not supported!".format(attn_type))


    def forward(self, x):
        xsize = x.size()
        if self.attn_type=="vanilla":
            x = rearrange(x, "b c t h w -> b (t h w) c")
            x = checkpoint(self.attention, self.use_checkpoint, x)
            x = rearrange(x, "b (t h w) c -> b c t h w", t=xsize[-3], h=xsize[-2], w=xsize[-1])
        elif self.attn_type=="factorized":
            x = rearrange(x, "b c t h w -> b t (h w) c")
            x = checkpoint(self.attention_sp, self.use_checkpoint, x)
            x = rearrange(x, "b t (h w) c -> b h w t c", h=xsize[-2], w=xsize[-1])
            x = checkpoint(self.attention_t, self.use_checkpoint, x)
            x = rearrange(x, "b h w t c -> b c t h w")
        elif self.attn_type=="spatial":
            x = rearrange(x, "b c t h w -> b t (h w) c")
            x = checkpoint(self.attention, self.use_checkpoint, x)
            x = rearrange(x, "b t (h w) c -> b c t h w", h=xsize[-2], w=xsize[-1])
        elif self.attn_type=="temporal":
            x = rearrange(x, "b c t h w -> b h w t c")
            x = checkpoint(self.attention, self.use_checkpoint, x)
            x = rearrange(x, "b h w t c -> b c t h w")
        return x
