import torch
import torch.nn as nn

USE_DEEPSPEED = False

#TODO take care of DeepSpeed later!!! (clone does not work for deepspeed zero3)
class EMA(nn.Module):
    def __init__(self, model, smoothing_factor: float=0.999):
        super().__init__()
        if smoothing_factor<0 or smoothing_factor>1.0:
            raise ValueError("Smoothing factor must be between 0 and 1")
        self.register_buffer("smoothing_factor", torch.ones(1)*smoothing_factor)

        #create a copy of the model parameters
        for n,p in model.named_parameters():
            if p.requires_grad:
                newname = n.replace(".", "")
                self.register_buffer(newname, p.detach().clone().data) #register so we can store it

        #a placeholder for the params we use for backupping the original weights
        self.model_backup = []

    def forward(self, model):
        #update ema weights
        for n,p in model.named_parameters():
            if p.requires_grad:
                newname = n.replace(".", "")
                ema_param = self.get_buffer(newname)
                ema_param.data.copy_(ema_param*self.smoothing_factor + p.data*(1.0-self.smoothing_factor))

    def save(self, model):
        #save the current model weights so we do not loose them
        self.model_backup = [p.clone() for p in model.parameters()]

    def use_ema(self, model):
        #replace model weights with our ema weights
        for n,p in model.named_parameters():
            if p.requires_grad:
                newname = n.replace(".", "")
                p.data.copy_(self.get_buffer(newname))

    def backup(self, model):
        #put old weights back into place
        for p_current, p_backup in zip(self.model_backup, model.parameters()):
            p_current.data.copy_(p_backup)
