import abc
import torch
import torch.nn as nn

from ca_diffusion.modules.utils import shape_val

class Diffusor(nn.Module,abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def encode(self, x, t, eps=None):
        return x


class FlowMatching(Diffusor):
    def __init__(self, sigma_min=0.0, noise_prior="independent"):
        super().__init__()

        self.sigma_min = sigma_min
        self.noise_prior = noise_prior

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

        xt = (1.0-t)*x + (self.sigma_min+(1.0-self.sigma_min)*t)*eps

        return xt, eps