import torch
import torch.nn as nn

#TODO take care of DeepSpeed later!!!
class EMA(nn.Module):
    def __init__(self, model, decay: float=0.999):
        super().__init__()
        if decay<0 or decay>1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.decay = decay

    def forward(self):
        pass
