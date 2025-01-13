import os
from omegaconf import DictConfig, OmegaConf
import yaml

import torch
from lightning.pytorch import Trainer, seed_everything

import hydra
from hydra.utils import instantiate

import ca_diffusion.flags as flags
from ca_diffusion.datamodule import DataModule

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    #ok save config -> one config is already saved in .hydra!!!
    #save_cfg = OmegaConf.to_container(cfg, resolve=True)
    #with open(os.path.join(cfg.paths.output_dir, "config.yaml"), "w") as f:
    #    yaml.dump(save_cfg, f)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility

    flags.init_flags(cfg.flags)
    #instantiate model we want to train
    model = instantiate(OmegaConf.to_container(cfg.model))

    #instantiate dataset we use for training and maybe validation
    datamodule = DataModule(**cfg.data)
    datamodule.setup()


    #set up trainer
    trainer_kwargs = instantiate(cfg.trainer)
    #trainer requires list instead of configs for some arguments
    #-> remove and convert them and manually add to trainer
    callbacks = trainer_kwargs.pop("callbacks", None)
    if isinstance(callbacks, DictConfig):
        callbacks = [callbacks[k] for k in callbacks.keys()]
    logger = trainer_kwargs.pop("logger", None)
    if isinstance(logger, DictConfig):
        logger = [logger[k] for k in logger.keys()]
    plugins = trainer_kwargs.pop("plugins", None)
    if isinstance(plugins, DictConfig):
        plugins = [plugins[k] for k in plugins.keys()]

    trainer = Trainer(callbacks=callbacks, logger=logger, plugins=plugins, **trainer_kwargs)

    #check if we are using deepspeed -> if yes, patch checkpointing

    #start training
    trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path", None))

##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    main()
