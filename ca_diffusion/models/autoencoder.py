import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

from ca_diffusion.modules.ema import EMA

#TODO: maybe put this in a separate file to support multiple distributions
#Gaussian distribution used for VAE
class GaussianDistribution():
    def __init__(self, z, deterministic=False, dim=1):
        self.deterministic = deterministic
        self.data = z
        if deterministic:
            self.mean = z
        else:
            self.mean, self.logvar = torch.chunk(z, 2, dim=dim)

    def sample(self, noise=None):
        if self.deterministic:
            return self.mean
        else:
            if noise is None:
                noise = torch.randn_like(self.mean)
            return self.mean + torch.exp(self.logvar*0.5)*noise
    
    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0]).to(self.data.device)
        else:
            if other is None:
                 kl = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
                 return kl/float(self.data.size(0))
            else:
                kl = 0.5 * torch.sum(torch.pow(self.mean-other.mean, 2)/other.var + self.var/other.var - 1.0 - self.logvar + other.logvar)
                return kl/float(self.data.size(0))
            

#TODO: add EMA
class Autoencoder(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 deterministic_z: bool=True,
                 noise_shape=(1,64,64),
                 image_mode: bool=True,
                 data_key: str="image",
                 ema_update_steps: int=-1,
                 ema_smoothing_factor: float=0.999,
                 ignore_param: list=[],
                 ckpt_path: str=None,
                 ignore_ckpt_keys: list=[]
                 ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder"]+ignore_param, logger=False) #do net hyperparams to logger

        self.deterministic_z = deterministic_z
        self.noise_shape = noise_shape
        self.image_mode = image_mode
        self.data_key = data_key
        self.ema_update_steps = ema_update_steps

        self.encoder = encoder
        self.decoder = decoder
        self.decode_args = {}

        #set up EMA if necessary
        self.use_ema = self.ema_update_steps>0
        if self.use_ema:
            self.encoder_ema = EMA(self.encoder, ema_smoothing_factor)
            self.decoder_ema = EMA(self.decoder, ema_smoothing_factor)

        if ckpt_path is not None and os.path.exists(ckpt_path):
            self.init_from_ckpt(ckpt_path, ignore_ckpt_keys)

    def init_from_ckpt(self, ckpt_path, ignore_keys: list=[]):
        """
        Initialize weights only from a checkpoint file 
        """
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        for k in sd.keys():
            for k2 in ignore_keys:
                if k.startswith(k2):
                    print("Removing key {} from state_dict".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        if len(missing)>0:
            print("Missing keys:", missing)
        if len(unexpected)>0:
            print("Unexpected keys:", unexpected)

    @contextmanager
    def ema(self, desc=None):
        if self.use_ema:
            #save weights for later and use ema weights
            self.encoder_ema.save(self.encoder)
            self.encoder_ema.use_ema(self.encoder)
            self.decoder_ema.save(self.decoder)
            self.decoder_ema.use_ema(self.decoder)
            if desc is not None:
                print("{}: Entering EMA!".format(desc))

        try:
            yield None
        finally:
            if self.use_ema:
                self.encoder_ema.backup(self.encoder)
                self.decoder_ema.backup(self.decoder)
                if desc is not None:
                    print("{}: Leaving EMA!".format(desc))

    def encode(self, x):
        z = self.encoder(x)
        return GaussianDistribution(z, deterministic=self.deterministic_z)

    def decode(self, z, noise=None):
        z = z.sample(noise)
        return self.decoder(z)
    
    def forward(self, x):
        dist = self.encode(x)
        rec = self.decode(dist)
        return rec
    
    def training_step(self, batch, batch_idx):
        log = {}

        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        loss = F.mse_loss(rec, batch[self.data_key], reduction="mean") #/float(z.size(0))
        log["train/rec_loss"] = loss.item()
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss    
    
    def validation_step(self, batch, batch_idx):
        log = {}
        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        loss = F.mse_loss(rec, batch[self.data_key], reduction="mean") #/float(z.size(0))
        log["val/rec_loss"] = loss.item()
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    #uncomment to search for unused parameters or inspect gradients
    #def on_after_backward(self):
    #    for name, param in self.named_parameters():
    #        if param.grad is None:
    #            print(name)

    def on_before_zero_grad(self, *args, **kwargs):
        if self.use_ema:
            if self.global_step%self.ema_update_steps==0 and self.global_step>1: #update ema every n steps
                self.encoder_ema(self.encoder)
                self.decoder_ema(self.decoder)

    def configure_optimizers(self):
        #TODO: split params in groups with and without weight decay!
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                             "interval": "step",
                                                             "frequency": 1}}
        return {"optimizer": optimizer}

    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = {}
        gt = batch[self.data_key]
        with self.ema("Logging"): #use EMA weights for logging images
            z = self.encode(gt)

            if self.image_mode:
                samples = [gt.cpu().unsqueeze(2)]
                for i in range(10):
                    sample = self.decode(z, **self.decode_args)
                    samples.append(sample.cpu().unsqueeze(2))

                samples.append(torch.mean(torch.cat(samples[1:], dim=2),dim=2).unsqueeze(2))
                log["samples"] = torch.cat(samples, dim=2)
            else:
                sample = self.decode(z, **self.decode_args)
                gt = torch.cat([torch.mean(gt, dim=2).unsqueeze(2), gt], dim=2).cpu()
                sample = torch.cat([torch.mean(sample, dim=2).unsqueeze(2), sample], dim=2).cpu()

                T = min(16, gt.size(2))
                log["samples"] = torch.cat([gt[:,:,:T], sample[:,:,:T]], dim=-2)
        return log
    

class AutoencoderGAN(Autoencoder):
    def __init__(self,
                 discriminator: nn.Module,
                 gan_weight: 1.0,
                 **kwargs):
        super().__init__(ignore_param=["discriminator"], **kwargs)

        self.discriminator = discriminator
        self.gan_weight = gan_weight

        self.automatic_optimization = False

    #TODO: change manual gradient accumulation N=4
    def training_step(self, batch, batch_idx):
        log = {}

        optim_g, optim_d = self.optimizers()

        #update generator
        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        pred_fake = self.discriminator(rec)

        loss_rec = F.mse_loss(rec, batch[self.data_key], reduction="mean") #/float(z.size(0))
        loss_g = torch.mean(-pred_fake)
        
        loss = loss_rec + loss_g*self.gan_weight
        self.manual_backward(loss)
        self.clip_gradients(optim_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        if (batch_idx + 1) % 4 == 0: #do gradient accumulation
            optim_g.step()
            #maybe run EMA manually here!!!
            optim_g.zero_grad()

        log["train/loss_ae_rec"] = loss_rec.item()
        log["train/loss_ae_gan"] = loss_g.item()
        log["train/loss_ae"] = loss.item()

        #update discriminator
        pred_real = self.discriminator(batch[self.data_key])
        pred_fake = self.discriminator(rec.detach())
        loss_real = torch.mean(F.relu(1.0-pred_real))
        loss_fake = torch.mean(F.relu(1.0+pred_fake))
        loss = (loss_real+loss_fake)*0.5
        self.manual_backward(loss)
        self.clip_gradients(optim_d, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        if (batch_idx + 1) % 4 == 0: #do gradient accumulation
            optim_d.step()
            optim_d.zero_grad()

        log["train/loss_d_real"] = loss_real.item()
        log["train/loss_d_fake"] = loss_fake.item()
        log["train/loss_d"] = loss.item()

        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer_g = self.hparams.optimizer(params=params)
        optimizer_d = self.hparams.optimizer(params=self.discriminator.parameters())
        if self.hparams.scheduler is not None:
            scheduler_g = self.hparams.scheduler(optimizer=optimizer_g)
            scheduler_d = self.hparams.scheduler(optimizer=optimizer_d)

            return [{"optimizer": optimizer_g, 
                    "lr_scheduler": {"scheduler": scheduler_g, "interval": "step", "frequency": 1}},
                    {"optimizer": optimizer_d, 
                    "lr_scheduler": {"scheduler": scheduler_d, "interval": "step", "frequency": 1}}]
        return [{"optimizer": optimizer_g}, {"optimizer": optimizer_d}]
    


class DiffusionAutoencoder(Autoencoder):
    def __init__(self,
                 diffusor: nn.Module,
                 **kwargs):
        super().__init__(ignore_param=["diffusor"], **kwargs)

        self.diffusor = diffusor
        self.decode_args = {"strategy": "euler",
                            "num_steps": 50}
    
    def decode(self, z, noise=None, **sample_args):
        model_args={"inj": z.sample()}
        if noise is None:
            noise = self.diffusor.sample_noise([z.data.size(0)]+list(self.noise_shape)).to(self.device)
        sample = self.diffusor.sample(self.decoder, noise, model_args=model_args, **sample_args)
        return sample

    def training_step(self, batch, batch_idx):
        log = {}

        z = self.encode(batch[self.data_key])
        model_args = {"inj": z.sample()}
        loss, loss_dict, data_dict = self.diffusor(self.decoder, batch[self.data_key], model_args=model_args)
        for k, v in loss_dict.items():
            log["train/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        log = {}
        z = self.encode(batch[self.data_key])
        model_args = {"inj": z.sample()}
        loss, loss_dict, data_dict = self.diffusor(self.decoder, batch[self.data_key], model_args=model_args)
        for k, v in loss_dict.items():
            log["val/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.diffusor.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler,
                                                             "interval": "step",
                                                             "frequency": 1}}
        return {"optimizer": optimizer}

