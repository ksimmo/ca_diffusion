import torch
import torch.nn as nn

def shape_val(x, y, pos=-1):
    """
    adds channels at pos such that dim(x) matches dim(y)
    """
    while len(x.size())<len(y.size()):
            x = x.unsqueeze(pos)
    return x

def checkpoint(function, use_checkpoint=True, *args, **kwargs):
    if use_checkpoint:
        return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)

#taken from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/models/image_transformer_v2.py
def zero_init(layer):
    """
    Zero initialize this layer
    """
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

def tag_param(param, tag):
    if not hasattr(param, "_tags"):
        param._tags = set([tag])
    else:
        param._tags.add(tag)
    return param

def apply_wd(module):
    """
    Tag this module for applying weight decay
    """
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            tag_param(param, "wd")
    return module