import os
from omegaconf import DictConfig, OmegaConf

import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything

import webdataset as wds
import time

import hydra
from hydra.utils import instantiate

from ca_diffusion.datamodule import DataModule


class Sharder(pl.LightningModule):
    def __init__(self, model, ckpt_path, output_path=".", name="name", prefix="train", max_count=1e6, max_size=1.0):
        super().__init__()

        self.output_path = output_path
        self.name = name
        self.prefix = prefix
        self.max_count = max_count
        self.max_size = max_size

        #initialize model
        self.model = model
        sd = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(sd["state_dict"], strict=True) 

        self.counter = 0

    def on_test_start(self):
        if self.global_rank==0:
            os.makedirs(os.path.join(self.output_path, self.name), exist_ok=True)
        time.sleep(5) #wait a bit so everything is set up
        self.writer = wds.ShardWriter(os.path.join(self.output_path, self.name, "{}_{}-%06d.tar".format(self.global_rank, self.prefix)), maxcount=self.max_count, maxsize=self.max_size*1e9) #-> each shard will be 1GB
        self.counter = 0


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        result = self.model.precompute(batch)

        keys = list(batch.keys())
        bs = len(batch[keys[0]]) #just get the batch size
        for b in range(bs):
            save_dict = {}
            save_dict["__key__"] = "{}_{:09d}".format(self.global_rank, self.counter) #identifier

            for k,v in result.items():
                if isinstance(v, torch.Tensor):
                    save_dict[k+".npy"] = v[b].cpu().numpy()
                #TODO: handle more datatypes or names (for instance class) here
                else:
                    print("Handling of type {} is not supported while sharding!".format(type(v))) #for now do not raise and error and just continue sharding

            #write dict to shard
            self.writer.write(save_dict)
            self.counter += 1

    def on_test_end(self):
        self.writer.close()



@hydra.main(config_path="configs", config_name="train")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    seed_everything(cfg.get("seed", 42), workers=True) #set random seed for reproducibility

    #instantiate model we want to train
    model = instantiate(OmegaConf.to_container(cfg.model))
    sharder = Sharder(model, cfg.ckpt_path, cfg.get("output_path", "shards"))

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

    trainer.test(sharder, datamodule)

##########################
if __name__ == "__main__":
    #configure torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True

    main()