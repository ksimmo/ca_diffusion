import abc

import numpy as np
import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class Diffusor(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, x, t, eps=None):
        return x


class FlowMatching(Diffusor):
    def __init__(self, sigma_min=0.0, timestep_sampling="linear", noise_prior="independent", learn_logvar=False):
        super().__init__()

        self.timestep_sampling = timestep_sampling
        self.sigma_min = sigma_min
        self.noise_prior = noise_prior
        self.learn_logvar = learn_logvar

        self.criterion = nn.MSELoss(reduction="none")

    def sample_noise(self, shape):
        if self.noise_prior=="independent":
            eps = torch.randn(shape)
        elif self.noise_prior=="mixed":
            eps = torch.randn(shape)
            eps = (eps+torch.randn_like(eps[:,:,:1]))*0.5
        elif self.noise_prior=="heavy_tailed":
            eps = torch.randn(shape)
            dist = torch.distributions.chi2.Chi2(self.chi_dof)
            k = dist.sample((eps.size(0),))
            k = torch.sqrt(k)
            k = shape_val(k, eps, -1)
            eps = eps/k
        else:
            raise NotImplementedError("Noise prior {} is not supported!".format(self.noise_prior))

        return eps

    def encode(self, x, t, eps=None):
        eps = self.sample_noise(x.size()).to(x.device) if eps is None else eps
        t = shape_val(t, x, -1)

        xt = (1.0-(1.0-self.sigma_min)*t)*x + t*eps

        return xt, eps
    
    def forward(self, model, x, t=None, loss_mask=None, **model_args):
        if t is None:
            if self.timestep_sampling=="linear":
                t = torch.rand((x.size(0),), device=x.device) #uniform sampling of t
            elif self.timestep_sampling=="logit_sigmoid":
                t = torch.sigmoid(torch.randn((x.size(0),), device=x.device))
            elif self.timestep_sampling=="mode":
                u = torch.randn((x.size(0),), device=x.device)
                s = 0.81
                t = 1.0-u-s*(torch.cos(np.pi*0.5*u).pow(2)-1+u)
            else:
                raise NotImplementedError("Timestep sampling method {} is not implemented!".format(self.timestep_sampling))
        xt, eps = self.encode(x, t)

        target = eps-(1.0-self.sigma_min)*x

        pred = model(x, t, **model_args)

        if loss_mask is None:
            val = torch.mean(self.criterion(pred, target), dim=np.arange(1,len(target.size())).tolist())
        else:
            val = torch.sum(self.criterion(pred, target)*loss_mask, dim=np.arange(1,len(target.size())).tolist())/(torch.sum(loss_mask, dim=np.arange(1,len(target.size())).tolist())+1e-12)

        if self.learn_logvar:
            logvar_t = self.logvar_net(t).squeeze(-1)
        else:
            logvar_t = torch.zeros_like(val)

        loss = torch.mean(val/torch.exp(logvar_t)+logvar_t)

        loss_dict = {}
        loss_dict["fm_loss"] = loss.item()

        return loss, loss_dict