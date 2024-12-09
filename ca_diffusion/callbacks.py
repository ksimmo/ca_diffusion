import os
import time

import numpy as np

import torch
from torchvision.utils import make_grid
from torchvision.io import write_png

from einops import rearrange

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.strategy.root_device.index)
        max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


#image logger
class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, disabled=False,
                 log_images_kwargs=None, log_val=True, log_val_epochs=1, save_local=True, save_logger=True,
                 compression_level=6):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_val = log_val
        self.compression_level = compression_level

        self.has_val_logged = False
        self.log_val_epochs = log_val_epochs
        self.save_local = save_local
        self.save_logger = save_logger
        
        self.train_steps = 0
        self.val_steps = 0

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            imgs = images[k]

            if len(imgs.size()) == 5:
                nrow = imgs.size(2)
                imgs = rearrange(imgs, "b c t h w -> (b t) c h w")
            elif len(imgs.size()) == 4:
                nrow = self.max_images

            grid = make_grid(imgs, nrow=nrow)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:

            imgs = images[k]

            if len(imgs.size()) == 5: #if we have saptiotemporal data -> unroll
                nrow = imgs.size(2)
                imgs = rearrange(imgs, "b c t h w -> (b t) c h w")
            elif len(imgs.size()) == 4:
                nrow = self.max_images

            grid = make_grid(imgs, nrow=nrow)
            
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid*255.0
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)

            write_png(grid.byte(), path, self.compression_level)
            

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx
        if split=="train":
            check_idx = self.train_steps
        elif split=="val":
            check_idx = self.val_steps
        else:
            check_idx = pl_module.global_step

        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            #make batch small to save GPU
            for k in batch.keys():
                b = min(len(batch[k]), self.max_images)
                batch[k] = batch[k][:b]

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)

            if self.save_local:
                self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            if self.save_logger:
                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if (check_idx % self.batch_freq) == 0 and (check_idx > 0):
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and (pl_module.global_step > 0):
            self.log_img(pl_module, batch, batch_idx, split="train")
        self.train_steps += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and pl_module.global_step > 0 and self.log_val and pl_module.current_epoch%self.log_val_epochs==0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        self.val_steps += 1
