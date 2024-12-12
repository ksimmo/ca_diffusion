import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

class Autoencoder(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 image_mode: bool=True,
                 ignore_param: list=[],
                 data_key: str="image"):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder"]+ignore_param, logger=False) #do net hyperparams to logger

        self.image_mode = image_mode
        self.data_key = data_key

        self.encoder = encoder
        self.decoder = decoder
        self.decode_args = {}

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def training_step(self, batch, batch_idx):
        log = {}

        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        loss = F.mse_loss(rec, batch[self.data_key], reduction="mean")
        log["train/rec_loss"] = loss.item()
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        log = {}
        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        loss = F.mse_loss(rec, batch[self.data_key], reduction="mean")
        log["val/rec_loss"] = loss.item()
        self.log_dict(log, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    #uncomment to search for unused parameters or inspect gradients
    """
    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)
            #else:
            #    print(name, torch.mean(param.grad))
    """

    def configure_optimizers(self):
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
                 **kwargs):
        super().__init__(ignore_param=["discriminator"], **kwargs)

        self.discriminator = discriminator

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        log = {}

        optim_g, optim_d = self.optimizers()

        #update generator
        self.toggle_optimizer(optim_g)
        optim_g.zero_grad()
        z = self.encode(batch[self.data_key])
        rec = self.decode(z)
        pred_fake = self.discriminator(rec)

        loss_rec = F.mse_loss(rec, batch[self.data_key], reduction="mean")
        loss_g = torch.mean(-pred_fake)
        
        loss = loss_rec + loss_g
        self.manual_backward(loss)
        optim_g.step()
        self.untoggle_optimizer(optim_g)
        log["train/loss_ae_rec"] = loss_rec.item()
        log["train/loss_ae_gan"] = loss_g.item()
        log["train/loss_ae"] = loss.item()

        #update discriminator
        self.toggle_optimizer(optim_d)
        optim_d.zero_grad()

        pred_real = self.discriminator(batch[self.data_key])
        pred_fake = self.discriminator(rec.detach())
        loss_real = torch.mean(F.relu(1.0-pred_real))
        loss_fake = torch.mean(F.relu(1.0+pred_fake))
        loss = (loss_real+loss_fake)*0.5
        self.manual_backward(loss)
        optim_d.step()
        self.untoggle_optimizer(optim_d)
        log["train/loss_d_reak"] = loss_real.item()
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

            return {"optimizer": [optimizer_g, optimizer_d], 
                    "lr_scheduler": [{"scheduler": scheduler_g, "interval": "step", "frequency": 1},
                                     {"scheduler": scheduler_d, "interval": "step", "frequency": 1}]}
        return {"optimizer": [optimizer_g, optimizer_d]}
    


class DiffusionAutoencoder(Autoencoder):
    def __init__(self,
                 diffusor: nn.Module,
                 noise_shape=(1,64,64),
                 **kwargs):
        super().__init__(ignore_param=["diffusor"], **kwargs)

        self.noise_shape = noise_shape
        self.diffusor = diffusor
        self.decode_args = {"strategy": "euler",
                            "num_steps": 50}

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z, noise=None, **sample_args):
        model_args={"inj": z}
        if noise is None:
            noise = self.diffusor.sample_noise([z.size(0)]+list(self.noise_shape)).to(self.device)
        sample = self.diffusor.sample(self.decoder, noise, model_args=model_args, **sample_args)
        return sample

    def training_step(self, batch, batch_idx):
        log = {}

        z = self.encode(batch[self.data_key])
        model_args = {"inj": z}
        loss, loss_dict, data_dict = self.diffusor(self.decoder, batch[self.data_key], model_args=model_args)
        for k, v in loss_dict.items():
            log["train/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        log = {}
        z = self.encode(batch[self.data_key])
        model_args = {"inj": z}
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

