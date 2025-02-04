import abc

import numpy as np
import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class LogvarNet(nn.Module):
    def __init__(self, channels_freq, learnable=False):
        super().__init__()

        if learnable:
            self.freqs = nn.Parameter(torch.randn((1,channels_freq)))
            self.phases = nn.Parameter(torch.rand((1,channels_freq)))
        else:
            self.register_buffer("freqs", torch.randn((1,channels_freq)))
            self.register_buffer("phases", torch.rand((1,channels_freq)))

        self.embedder = nn.Linear(channels_freq, 1)

    def forward(self, t):
        freqs = shape_val(self.freqs.to(torch.float32), t, pos=0)
        phases = shape_val(self.phases.to(torch.float32), t, pos=0)
        emb = torch.cos((freqs*t.to(torch.float32).unsqueeze(-1)+phases)*2.0*np.pi) #*np.sqrt(2.0) #<- magnitude preserving
        return self.embedder(emb.to(t.dtype))

######################################################
## Base Class

class Diffusor(nn.Module):
    def __init__(self, 
                 noise_prior: str="independent", 
                 timestep_sampling: str="linear",
                 learn_logvar: bool=False,
                 channels_logvar: int=64,
                 threshold_method: str="none",
                 threshold_args: dict={},
                 **kwargs
                 ):
        super().__init__()

        self.noise_prior = noise_prior
        self.timestep_sampling = timestep_sampling
        self.threshold_method = threshold_method
        self.threshold_args = threshold_args
        self.learn_logvar = learn_logvar
        if self.learn_logvar:
            self.logvar_net = LogvarNet(channels_freq=channels_logvar, learnable=True)

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
    
    def sample_t(self, B: int, device=None):
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
    
    #thresholding to prevent accumulation of errors during sampling
    def threshold(self, x, percentile=0.995, static_start=-1.0, static_end=1.0):
        if self.threshold_method=="none":
            pass
        elif self.threshold_method=="static":
            x = x.clamp(static_start, static_end)
        elif self.threshold_method=="dynamic":
            s = torch.quantile(torch.abs(x).view(x.size(0),-1), percentile, dim=-1)
            x = torch.clamp(x, -s,s)/s
        else:
            raise NotImplementedError("Thresholding method {} is not supported!".format(self.threshold_method))
        return x
    
    #overwrite regions with gt which were given by mask
    def end_of_sampling(self, noise, mask=None, gt=None, **kwargs):
        if mask is not None and gt is not None:
            noise = noise*(1.0-mask)+gt*mask
        return noise

######################################################
## Flow Matching


class FlowMatching(Diffusor):
    def __init__(self, sigma_min: float=0.0, **kwargs):
        super().__init__(**kwargs)

        self.sigma_min = sigma_min

    def encode(self, x, t, eps=None):
        eps = self.sample_noise(x.size()).to(x.device) if eps is None else eps
        t = shape_val(t, x, -1)
        xt = (1.0-(1.0-self.sigma_min)*t)*x + t*eps #eq. (22) [https://arxiv.org/pdf/2210.02747]
        return xt, eps
    
    def forward(self, model, x, t=None, loss_mask=None, model_args={}, return_data: bool=False):
        if t is None:
            t = self.sample_t(x.size(0), x.device)
        xt, eps = self.encode(x, t)

        target = eps-(1.0-self.sigma_min)*x #eq. (23) [https://arxiv.org/pdf/2210.02747]

        pred = model(xt, t, **model_args)

        #loss calculation
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

            noise = self.threshold(noise, **self.threshold_args)

        return noise
    
    def sample_midpoint(self, model, noise, num_steps=50, model_args={}):
        ts = torch.linspace(1.0, self.sigma_min, num_steps).to(noise.device)

        for step in range(1,num_steps):
            t_now = torch.full((noise.size(0),), ts[step], device=noise.device)
            t_prev = torch.full((noise.size(0),), ts[step-1], device=noise.device)

            dt = t_now-t_prev
            dt = shape_val(dt, noise)

            pred = model(noise, t_prev, **model_args)

            #correct
            intermediate = noise + pred*dt*0.5
            pred = model(intermediate, t_prev+(t_now-t_prev)*0.5, **model_args)

            #update
            noise = noise + pred*dt

            noise = self.threshold(noise, **self.threshold_args)

        return noise
    
    @torch.no_grad()
    def sample(self, model, noise, strategy: str="euler", **kwargs):
        if strategy=="euler":
            sample = self.sample_euler(model, noise, **kwargs)
        elif strategy=="midpoint":
            sample = self.sample_midpoint(model, noise, **kwargs)
        #elif strategy=="heun":
        #    sample = self.sample_heun(model, noise, **kwargs)
        #elif strategy=="rk4":
        #    sample = self.sample_rk4(model, noise, **kwargs)
        else:
            raise NotImplementedError("Sampling strategy {} is not supported!".format(strategy))
        
        return self.end_of_sampling(sample)
        
######################################################
## DDPM

class DDPMSchedule(nn.Module): #nn.Module in case we want to make the schedule learnable
    def __init__(self, 
                 enforce_0_snr: bool=False, 
                 snr_scale: float=1.0, 
                 **kwargs):
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
    def __init__(self, 
                 linear_start: float=1e-4, 
                 linear_scale: float=10.0,  
                 **kwargs
                 ):
        self.linear_start = linear_start
        self.linear_scale = linear_scale

        super().__init__(**kwargs)

    def func(self, t):
        return torch.exp(-self.linear_start-self.linear_scale*t.pow(2))
        
#continous DDPM implementation
class DDPM(Diffusor):
    def __init__(self, 
                 schedule="linear", 
                 schedule_args={}, 
                 parameterization="eps", 
                 **kwargs
                 ):
        super().__init__(**kwargs)

        assert parameterization in ["eps", "x0", "v"]
        self.parameterization = parameterization

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

        loss_dict = {}
        loss_dict["ddpm_loss"] = loss.item()

        data_dict = {}
        if return_data:
            data_dict["eps"] = eps

        return loss, loss_dict, data_dict
    
    def convert_eps_to_x0(self, pred, xprev, gamma):
        return (xprev-torch.sqrt(1.0-gamma)*pred)/torch.sqrt(gamma) #eq. (12) [https://arxiv.org/abs/2010.02502]

    def convert_v_to_eps(self, pred, xprev, gamma):
        return torch.sqrt(gamma)*pred+torch.sqrt(1.0-gamma)*xprev #convert v prediction to eps eq. (17) [https://arxiv.org/abs/2305.08891]
    
    def convert_v_to_x0(self, pred, xprev, gamma):
        return torch.sqrt(gamma)*xprev-torch.sqrt(1.0-gamma)*pred #convert v prediction to x0 eq. (19) [https://arxiv.org/abs/2305.08891]
    
    #TODO: check if this is correct
    def sample_ddim(self, model, noise, num_steps=50, eta=0.0, model_args={}):
        ts = torch.linspace(1.0, 0.0, num_steps).to(noise.device)

        for step in range(1,num_steps):
            t_now = torch.full((noise.size(0),), ts[step], device=noise.device)
            t_prev = torch.full((noise.size(0),), ts[step-1], device=noise.device)

            gamma_now = self.schedule(t_now)
            gamma_now = shape_val(gamma_now, noise, -1)
            gamma_prev = self.schedule(t_prev)
            gamma_prev = shape_val(gamma_prev, noise, -1)


            sigma_now = torch.sqrt((1.0-gamma_now)/(1.0-gamma_prev))*torch.sqrt(1.0-(gamma_prev/gamma_now))*eta #eq (3) [https://arxiv.org/abs/2010.02502]

            pred = model(noise, t_prev, **model_args)
            eps = torch.randn_like(noise)

            if self.parameterization=="eps":
                pred_x0 = self.convert_eps_to_x0(pred, noise, gamma_prev)
                noise = torch.sqrt(gamma_now)*pred_x0+torch.sqrt(1.0-gamma_now-sigma_now.pow(2))*pred+sigma_now*eps #eq. (12) [https://arxiv.org/abs/2010.02502]
            elif self.parameterization=="x0":
                pred_eps = self.convert_x0_to_eps(pred, noise, gamma_prev)
                noise = torch.sqrt(gamma_now)*pred+torch.sqrt(1.0-gamma_now-sigma_now.pow(2))*pred_eps+sigma_now*eps #eq. (12) [https://arxiv.org/abs/2010.02502]
            elif self.parameterization=="v":
                pred_x0 = self.convert_v_to_x0(pred, noise, gamma_prev)
                pred_eps = self.convert_v_to_eps(pred, noise, gamma_prev)
                noise = torch.sqrt(gamma_now)*pred_x0+torch.sqrt(1.0-gamma_now-sigma_now.pow(2))*pred_eps+sigma_now*eps #eq. (12) [https://arxiv.org/abs/2010.02502]

            noise = self.threshold(noise, **self.threshold_args)

        return noise

    @torch.no_grad()
    def sample(self, model, noise, strategy="ddim", **kwargs):
        #if strategy=="ddpm":
        #    sample = self.sample_ddpm(model, noise, **kwargs)
        #elif strategy=="ddim":
        #    sample = self.sample_ddim(model, noise, **kwargs)
        #else:
        raise NotImplementedError("Sampling strategy {} is not supported!".format(strategy))
    
        return self.end_of_sampling(sample)