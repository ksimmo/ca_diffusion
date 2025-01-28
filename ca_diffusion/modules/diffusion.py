import abc

import numpy as np
import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class Diffusor(nn.Module):
    def __init__(self, noise_prior="independent", timestep_sampling="linear"):
        super().__init__()

        self.noise_prior = noise_prior
        self.timestep_sampling = timestep_sampling

        self.criterion = nn.MSELoss(reduction="none") #TODO: maybe also support L1 here

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
    
    def sample_t(self, B, device=None):
        if self.timestep_sampling=="linear":
            t = torch.rand((B,), device=device) #uniform sampling of t
        elif self.timestep_sampling=="logit_sigmoid":
            t = torch.sigmoid(torch.randn((B,), device=device))
        elif self.timestep_sampling=="mode":
            u = torch.randn((B,), device=device)
            s = 0.81
            t = 1.0-u-s*(torch.cos(np.pi*0.5*u).pow(2)-1+u)
        else:
            raise NotImplementedError("Timestep sampling method {} is not implemented!".format(self.timestep_sampling))
        
        return t

    @abc.abstractmethod
    def encode(self, x, t, eps=None):
        return x


class FlowMatching(Diffusor):
    def __init__(self, sigma_min=0.0, noise_prior="independent", timestep_sampling="linear", learn_logvar=False):
        super().__init__(noise_prior=noise_prior, timestep_sampling=timestep_sampling)

        self.sigma_min = sigma_min
        self.learn_logvar = learn_logvar

    def encode(self, x, t, eps=None):
        eps = self.sample_noise(x.size()).to(x.device) if eps is None else eps
        t = shape_val(t, x, -1)
        xt = (1.0-(1.0-self.sigma_min)*t)*x + t*eps
        return xt, eps
    
    def forward(self, model, x, t=None, loss_mask=None, model_args={}, return_data=False):
        if t is None:
            t = self.sample_t(x.size(0), x.device)
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
        

class DDPMSchedule(nn.Module): #module incase we want to make the schedule learnable
    def __init__(self, enforce_0_snr=False, snr_scale=1.0, **kwargs):
        super().__init__()

        self.enforce_0_snr= enforce_0_snr
        self.snr_scale = snr_scale

    @abc.abstractmethod
    def func(self, t):
        "function to calculate gamma"

    def wrapped_call(self, t):
        if self.enforce_0_snr: #we need to do it dynamicall otherwise it can get numerically unstable at t=1.0 -> donno why
            with torch.no_grad():
                gamma_0 = self.func(torch.zeros_like(t))
                gamma_T = self.func(torch.ones_like(t))

                shift = gamma_T
                scale = gamma_0/(gamma_0-gamma_T)

            return (self.func(t)-shift)*scale #in contrast to [https://arxiv.org/abs/2305.08891] we do not rescale sqrt(abar) but abar instead
        else:
            return self.func(t)

    def forward(self, t):
        if self.snr_scale!=1.0: # solve snr*beta = (gamma*s)/(1-gamma*s) => s = beta/(1-gamma+gamma*beta)
            factor = self.snr_scale**2/(1.0-self.wrapped_call(t)+self.wrapped_call(t)*self.snr_scale**2) 
            return self.wrapped_call(t)*factor
        else:
            return self.wrapped_call(t) #for 1.0 we can just express it way simpler
        
class LinearSchedule(DDPMSchedule):
    def __init__(self, linear_start=1e-4, linear_scale=10.0,  **kwargs):
        self.linear_start = linear_start
        self.linear_scale = linear_scale

        super().__init__(**kwargs)

    def func(self, t):
        return torch.exp(-self.linear_start-self.linear_scale*t.pow(2))
        
#continous DDPM implementation
class DDPM(Diffusor):
    def __init__(self, schedule="linear", schedule_args={}, parameterization="eps", timestep_sampling="linear", noise_prior="independent", learn_logvar=False):
        super().__init__(noise_prior=noise_prior, timestep_sampling=timestep_sampling)

        self.timestep_sampling = timestep_sampling
        assert parameterization in ["eps", "x0", "v"]
        self.parameterization = parameterization
        self.learn_logvar = learn_logvar

        if schedule=="linear":
            self.schedule = LinearSchedule(**schedule_args)
        else:
            raise NotImplementedError("DDPM schedule {} is not supported!".format(schedule))

    def encode(self, x, t, eps=None):
        eps = self.sample_noise(x.size()).to(x.device) if eps is None else eps
        t = shape_val(t, x, -1)
        gamma = self.schedule(t)
        xt = torch.sqrt(gamma)*x + torch.sqrt(1.0-gamma)*eps
        return xt, eps, gamma
    
    def forward(self, model, x, t=None, loss_mask=None, model_args={}, return_data=False):
        if t is None:
            t = self.sample_t(x.size(0), x.device)
        xt, eps, gamma = self.encode(x, t)

        if self.parameterization=="eps":
            target = eps
        elif self.parameterization=="x0":
            target = x
        elif self.parameterization=="v":
            target = torch.sqrt(gamma)*eps - torch.sqrt(1.0-gamma)*x
        else:
            raise NotImplementedError("Parameterization {} is not supported!".format(self.parameterization))

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

    @torch.no_grad()
    def sample(self, model, noise, strategy="ddim", **kwargs):
        #if strategy=="ddpm":
        #    return self.sample_ddpm(model, noise, **kwargs)
        #elif strategy=="ddim":
        #    return self.sample_ddim(model, noise, **kwargs)
        #else:
        raise NotImplementedError("Sampling strategy {} is not supported!".format(strategy))