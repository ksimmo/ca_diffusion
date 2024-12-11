import torch
import torch.nn as nn

from ca_diffusion.modules.transformer.blocks import ConditionTransformerBlock
from ca_diffusion.modules.networks.diffusion.utils import TimestepEmbedder

class DiT(nn.Module):
    def __init__(self, channels_in, channels, channels_emb, channels_out=None, depth=8, num_heads=12, head_channels=None,
                 qk_norm=True, expand=4, dropout=0.0, compile=False, use_checkpoint=False, channels_freq=256):
        super().__init__()

        self.timestep_embedder = TimestepEmbedder(channels_freq, channels, learnable=True)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(ConditionTransformerBlock(channels, channels_emb, num_heads=num_heads, head_channels=head_channels, attn_dropout=dropout,
                                                         qk_norm=qk_norm, expand=expand, ff_dropout=dropout,
                                                         compile_attn=compile, compile_ff=compile, use_checkpoint=use_checkpoint))

    def forward(self, x, t):
        pass
