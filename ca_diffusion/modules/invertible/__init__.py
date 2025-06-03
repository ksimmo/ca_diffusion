from abc import abstractmethod

import torch
import torch.nn as nn

class InvertibleModule(nn.Module):
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        #Fill in
    
    @abstractmethod
    def backward(self, x, *args, **kwargs):
        #Fill in
    
class InvertibleSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        logdet = torch.zeros(1)
        for layer in self:
            x,l = layer.forward(x, *args, **kwargs)
            logdet += l
        return x, logdet
    
    def backward(sel, x, *args, **kwargs):
        length = len(self)
        for i in reversed(range(length)):
            x = self[i].backward(x, *args,**kwargs)
        return x
    
class InvertibleModuleList(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        logdet = torch.zeros(1)
        for layer in self:
            x,l = layer.forward(x, *args, **kwargs)
            logdet += l
        return x, logdet
    
    def backward(sel, x, *args, **kwargs):
        length = len(self)
        for i in reversed(range(length)):
            x = self[i].backward(x, *args,**kwargs)
        return x

