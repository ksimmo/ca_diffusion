import os

import random
import numpy as np


import torch
from torch.utils.data._utils.collate import default_collate
import lightning as pl


import webdataset as wds
import wids

from functools import partial

from hydra.utils import instantiate

def worker_init_fn(worker_id):
    """
    Function to seed all underlying libraries in the workers
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

####################################################
def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys} #remove keys with "__"

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.tensor(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.tensor(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])
    return result

#ok default unbatching in webdataset does not support dictionaries
def _unbatched(data):
    """
    Turn batched data back into unbatched data but keeps dictionaries
    """
    for sample in data:
        keys = list(sample.keys())
        bs = len(sample[keys[0]])
        assert len(sample) > 0
        for i in range(bs):
            yield tuple((k,v[i]) for k,v in sample.items()) #return a tuple of key value pairs 


unbatched = wds.filters.pipelinefilter(_unbatched)

#map key-value pairs back to dictionary
def unbatch_collation_fn(samples): #TODO: maybe there is a more efficient way
    """
    Collation function for unbatching which is working with dicts
    """
    result = {}
    for s in samples:
        for i in range(len(s)):
            if s[i][0] not in result.keys():
                result[s[i][0]] = [s[i][1]]
            else:
                result[s[i][0]].append(s[i][1])

    for k in result.keys():
        if isinstance(result[k][0], (int, float)):
            result[k] = torch.tensor(np.array(result[k]))
        elif isinstance(result[k][0], torch.Tensor):
            result[k] = torch.stack(result[k])
        elif isinstance(result[k][0], np.ndarray):
            result[k] = torch.tensor(np.array(result[k]))
    return result


def wids_collate_fn(samples):
    """
    Collation function for indexed webdataset
    """
    #ok check if we need to erase an element
    for i in reversed(range(len(samples))):
        if len(samples.keys())==0:
            del samples[i]
    #use normal collation from pytorch
    return torch.utils.data.default_collate(samples)


class DataModule(pl.LightningDataModule):
    def __init__(self, train=None, validation=None, test=None, **kwargs):
        super().__init__()

        self.dataset_configs = {"train": train, "validation": validation, "test": test}

    def setup(self): #create necessary datasets at the beginning
        self.datasets = {}
        for k in self.dataset_configs.keys():
            if self.dataset_configs[k] is not None:
                variant = self.dataset_configs[k].get("variant", None)
                if variant=="torch": #instantiate torch datasets for later use
                    if self.dataset_configs[k] is not None:
                        self.datasets[k] = instantiate(self.dataset_configs[k]["config"])

    def wds_check_keys(self, sample, keys):
        return set(keys).issubset(sample.keys())

    def get_loader(self, config, mode="train"):
        variant = config.get("variant", None)
        if variant=="torch":
            worker_init = config.get("worker_init", True)
            shuffle = config.get("shuffle", False)
            drop_last = config.get("drop_last", False)
            return torch.utils.data.DataLoader(self.datasets[mode], batch_size=config["batch_size"], num_workers=config["num_workers"], shuffle=shuffle, worker_init_fn=worker_init_fn if worker_init else None, drop_last=drop_last)
        elif variant=="wds": #webdataset
            #initialize shards
            if isinstance(config["tar_base"], str) and isinstance(config["tars"], str):
                shards = os.path.join(config["tar_base"], config["tars"])
            elif isinstance(config["tar_base"], (list, tuple)) and isinstance(config["tars"], (list, tuple)):
                shards = []
                for i in range(len(config["tar_base"])):
                    shards += wds.shardlists.expand_urls(os.path.join(config["tar_base"][i], config["tars"][i]))
            elif isinstance(config["tar_base"], str) and isinstance(config["tars"], (list, tuple)):
                shards = []
                for i in range(len(config["tars"])):
                    shards += wds.shardlists.expand_urls(os.path.join(config["tar_base"], config["tars"][i]))
            else:
                raise Exception("Invalid tar configuration!")
            
            resampled = config.get("resampled", False)
            multinode = config.get("multinode", False)
            shardshuffle = config.get("shardshuffle", False)
            empty_check = config.get("empty_check", True)
            shuffle = config.get("shuffle", 0)
            repeat = config.get("repeat", None)
            interprocess_shuffle = config.get("interprocess_shuffle", 0)
            epoch_length = config.get("epoch_length", None)

            dataset = wds.WebDataset(shards, resampled=resampled, 
                                        nodesplitter = wds.shardlists.split_by_node if multinode else wds.shardlists.single_node_only, shardshuffle=shardshuffle, handler=wds.warn_and_continue, 
                                        empty_check=empty_check).shuffle(shuffle)
            if repeat is not None:
                dataset = dataset.repeat(repeat)

            #call functions of webdataset type here
            interface = config.get("interface", None)
            if interface is not None:
                interface = instantiate(interface)
                dataset = interface.config_webdataset(dataset)
            else:
                dataset = dataset.decode()

            required_keys = config.get("required_keys", [])
            if len(required_keys)>0:
                dataset = dataset.select(partial(self.wds_check_keys, keys=required_keys))

            #dataset = dataset.shuffle(shuffle) #this takes some significant time from videos
            dataset = dataset.batched(config["batch_size"], collation_fn=default_collate) #dict_collation_fn) #partial=False

            loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=config["num_workers"])
            #shuffle accross processes
            if interprocess_shuffle>1:
                loader = loader.compose(unbatched()).shuffle(interprocess_shuffle).batched(config["batch_size"], collation_fn=unbatch_collation_fn) #partial=True
            if epoch_length is not None:
                #get number of gpus
                rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
                loader = loader.with_epoch(epoch_length//config["batch_size"]//world_size)
            return loader
        elif variant=="wids": #indexed webdataset
            dataset = wids.ShardListDataset(config["index_file"], keep=True, base=config["tar_base"])

            interface = config.get("interface", None)
            if interface is not None:
                interface = instantiate(interface)
                dataset = interface.config_wids(dataset)

            shuffle = config.get("shuffle", False)

            sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=shuffle)
            loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=config["num_workers"], worker_init_fn=worker_init_fn, collate_fn=wids_collate_fn)

            return loader
        else:
            raise NotImplementedError("Dataset format {} is not supported!".format(variant))
        
    def train_dataloader(self):
        return self.get_loader(self.dataset_configs["train"], "train")

    def val_dataloader(self):
        return self.get_loader(self.dataset_configs["validation"], "validation")

    def test_dataloader(self):
        return self.get_loader(self.dataset_configs["test"], "test")
