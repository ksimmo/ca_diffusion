import torch
import torch.nn as nn

import lightning.pytorch as pl

class DiffusionAutoencoder(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module, 
                 decoder: nn.Module,
                 diffusor: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler=None,
                 image_mode: bool=True):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "decoder", "diffusor"], logger=False) #do net hyperparams to logger

        self.image_mode = image_mode

        self.encoder = encoder
        self.decoder = decoder
        self.diffusor = diffusor

    def training_step(self, batch, batch_idx):
        log = {}

        z = self.encoder(batch["image"])
        model_args = {"inj": z}
        loss, loss_dict = self.diffusor(self.decoder, batch["image"], model_args=model_args)
        for k, v in loss_dict.items():
            log["train/"+k] = v
        self.log_dict(log, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    #uncomment to search for unused parameters or inspect gradients
    """
    def on_after_backward(self):
        for name, param in self.model.named_parameters():
            if param.grad is None:
                print(name)
            #else:
            #    print(name, torch.mean(param.grad))
    """
    
    def validation_step(self, batch, batch_idx):
        log = {}
        z = self.encoder(batch["image"])
        model_args = {"inj": z}
        loss, loss_dict = self.diffusor(self.decoder, batch["image"], model_args=model_args)
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
    
    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = {}
        gt = batch["image"]

        z = self.encoder(batch["image"])
        model_args = {"inj": z}

        if self.image_mode:
            samples = [gt.cpu().unsqueeze(2)]
            for i in range(10):
                sample = self.diffusor.sample(self.decoder, self.diffusor.sample_noise(gt.size()).to(self.device), "euler", model_args=model_args)
                samples.append(sample.cpu().unsqueeze(2))

            samples.append(torch.mean(torch.cat(samples[1:], dim=2),dim=2).unsqueeze(2))
            log["samples"] = torch.cat(samples, dim=2)
        else:
            sample = self.diffusor.sample(self.decoder, self.diffusor.sample_noise(gt.size()).to(self.device), "euler", model_args=model_args)
            gt = torch.cat([torch.mean(gt, dim=2).unsqueeze(2), gt], dim=2).cpu()
            sample = torch.cat([torch.mean(sample, dim=2).unsqueeze(2), sample], dim=2).cpu()

            log["samples"] = torch.cat([gt, sample], dim=-2)
        return log

