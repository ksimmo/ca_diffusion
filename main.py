
from omegaconf import DictConfig, OmegaConf

import torch
from lightning.pytorch import Trainer, seed_everything

import hydra
from hydra.utils import instantiate

@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility

    #instantiate model
    model = instantiate(cfg.model)

    #instantiate callbacks -> we can have multiple ones
    callbacks_cfg = cfg.get("callbacks", None)
    if callbacks_cfg is not None:
        if isinstance(callbacks_cfg, DictConfig):
            callbacks = []
            for k, v in callbacks_cfg.items():
                callbacks.append(instantiate(cfg.logger))
        else:
            callbacks = None

    #instantiate logger -> we can have multiple ones
    logger_cfg = cfg.get("logger", None)
    if logger_cfg is not None:
        if isinstance(logger_cfg, DictConfig):
            logger = []
            for k, v in logger_cfg.items():
                logger.append(instantiate(cfg.logger))
        else:
            logger = None

    #instantiate dataset
    datamodule = instantiate(cfg.data)
    datamodule.setup(stage="")

    trainer = Trainer(cfg.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(model, datamodule, ckpt_path=None)

##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    main()