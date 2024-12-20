import numpy as np

import torch
import torch.nn as nn

from einops import rearrange

from ca_diffusion.modules.norm import AdaRMSNorm
from ca_diffusion.modules.transformer.blocks import ConditionTransformerBlock
from ca_diffusion.modules.networks.diffusion.utils import TimestepEmbedder

class DiT(nn.Module):
    def __init__(self, channels_in, channels, channels_emb, channels_out=None, depth=8, num_heads=12, head_channels=None,
                 qk_norm=True, expand=4, dropout=0.0, compile=False, use_checkpoint=False, channels_freq=256,
                 patch_size=[1,1]):
        super().__init__()

        channels_out = channels_in if channels_out is None else channels_out
        self.patch_size = patch_size

        self.timestep_embedder = TimestepEmbedder(channels_freq, channels, learnable=True)

        self.proj_in = nn.Linear(channels_in*int(np.prod(patch_size)), channels)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConditionTransformerBlock(channels, channels_emb, num_heads=num_heads, head_channels=head_channels, attn_dropout=dropout,
                                                         qk_norm=qk_norm, expand=expand, ff_dropout=dropout,
                                                         compile_attn=compile, compile_ff=compile, use_checkpoint=use_checkpoint))
            
        self.final_norm = AdaRMSNorm(channels, channels_emb)
        self.proj_out = nn.Linear(channels, channels_out*int(np.prod(patch_size)))

    def forward(self, x, t):
        emb = self.timestep_embedder(t)

        #patchify
        if len(self.patch_size)==2:
            H = x.size(2)//self.patch_size[0]
            W = x.size(3)//self.patch_size[1]
            x = rearrange(x, "b c (h i) (w j) -> b (h w) (c i j)", i=self.patch_size[0], j=self.patch_size[1])
        elif len(self.patch_size)==3:
            T = x.size(2)//self.patch_size[0]
            H = x.size(3)//self.patch_size[1]
            W = x.size(4)//self.patch_size[2]
            x = rearrange(x, "b c (t i) (h j) (w k) -> b (t h w) (c i j k)", i=self.patch_size[0], j=self.patch_size[1], k=self.patch_size[2])

        x = self.proj_in(x)

        #add position

        for block in self.blocks:
            x = block(x, emb)
        x = self.final_norm(x, emb)
        x = self.proj_out(x)

        #unpatch
        if len(self.patch_size)==2:
            x = rearrange(x, "b (h w) (c i j) -> b c (h i) (w j)", i=self.patch_size[0], j=self.patch_size[1], h=H, w=W)
        elif len(self.patch_size)==3:
            x = rearrange(x, "b (t h w) (c i j k) -> b c (t i) (h j) (w k)", i=self.patch_size[0], j=self.patch_size[1], k=self.patch_size[2], h=H, w=W)

        return x