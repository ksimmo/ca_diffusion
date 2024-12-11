import abc

import numpy as np
import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class Diffusor(nn.Module):
    def __init__(self, noise_prior="independent"):
        super().__init__()

        self.noise_prior = noise_prior

    def sample_noise(self, shape):
        if self.noise_prior=="independent":
            eps = torch.randn(shape)
        elif self.noise_prior=="mixed": #use for spatio-temporal data only!!!
            eps = torch.randn(shape)
            eps = (eps+torch.randn_like(eps[:,:,:1]))*0.5
        elif self.noise_prior=="heavy_tailed": #https://arxiv.org/abs/2410.14171
            eps = torch.randn(shape)
            dist = torch.distributions.chi2.Chi2(self.chi_dof)
            k = dist.sample((eps.size(0),))
            k = torch.sqrt(k)
            k = shape_val(k, eps, -1)
            eps = eps/k
        else:
            raise NotImplementedError("Noise prior {} is not supported!".format(self.noise_prior))

        return eps

    @abc.abstractmethod
    def encode(self, x, t, eps=None):
        return x


class FlowMatching(Diffusor):
    def __init__(self, sigma_min=0.0, timestep_sampling="linear", noise_prior="independent", learn_logvar=False):
        super().__init__(noise_prior=noise_prior)

        self.timestep_sampling = timestep_sampling
        self.sigma_min = sigma_min
        self.learn_logvar = learn_logvar

        self.criterion = nn.MSELoss(reduction="none")

    def encode(self, x, t, eps=None):
        eps = self.sample_noise(x.size()).to(x.device) if eps is None else eps
        t = shape_val(t, x, -1)

        xt = (1.0-(1.0-self.sigma_min)*t)*x + t*eps

        return xt, eps
    
    def forward(self, model, x, t=None, loss_mask=None, model_args={}, return_data=False):
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

        pred = model(xt, t, **model_args)

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

        data_dict = {}
        if return_data:
            data_dict["eps"] = eps
            #data_dict["pred_x0"] = xt+pred*(1.0-self.sigma_min)

        return loss, loss_dict, data_dict

    def sample_euler(self, model, noise, num_steps=50, model_args={}):
        ts = torch.linspace(1.0, self.sigma_min, num_steps).to(noise.device)

        for step in range(1,num_steps):
            t_now = torch.full((noise.size(0),), ts[step], device=noise.device)
            t_prev = torch.full((noise.size(0),), ts[step-1], device=noise.device)

            dt = t_now-t_prev
            dt = shape_val(dt, noise)

            pred = model(noise, t_prev, **model_args)

            #update
            noise = noise + pred*dt

        return noise
    
    @torch.no_grad()
    def sample(self, model, noise, strategy="euler", **kwargs):
        if strategy=="euler":
            return self.sample_euler(model, noise, **kwargs)
        #elif strategy=="heun":
        #    return self.sample_heun(model, noise, **kwargs)
        #elif strategy=="midpoint":
        #    return self.sample_midpoint(model, noise, **kwargs)
        #elif strategy=="rk4":
        #    return self.sample_rk4(model, noise, **kwargs)
        else:
            raise NotImplementedError("Sampling strategy {} is not supported!".format(strategy))