import os

import copy

import torch
from torchvision.io import decode_image
import torchvision.transforms.v2 as ttf

class ImageCollector:
    def __init__(self, data_path, typ, split="train", **kwargs):
        if not isinstance(data_path, (list, tuple)):
            data_path = [data_path]
        if not isinstance(typ, (list, tuple)):
            typ = [typ]

        assert len(data_path)==len(typ), "Same number of data_paths and typs needs to be given!"

        self.split = split

        self.image_infos = []
        self.init_collector(data_path, typ, split, **kwargs)

    def init_collector(self, data_path, typ, split, **kwargs):
        for i in range(len(data_path)):
            ty = typ[i]
            dp = data_path[i]

            if ty=="image_folder":
                self.init_image_folder(dp, split, **kwargs)
            elif ty=="imagenet":
                self.init_imagenet(dp, split, **kwargs)
            else:
                raise NotImplementedError("Collecting images from {} dataset is not supported!".format(ty))
            
    def init_image_folder(self, data_path, split, **kwargs):
        for f in sorted(os.listdir(data_path)):
            self.image_infos.append({
                    "path": os.path.join(data_path, f)
                    })
            
    def init_imagenet(self, data_path, split, **kwargs):
        path = os.path.join(data_path, "ILSVRC2012_"+split)

        for d in sorted(os.listdir(os.path.join(path, "data"))):
            for f in sorted(os.listdir(os.path.join(path, "data", d))):
                self.image_infos.append({
                    "path": os.path.join(path, "data", d, f),
                })



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, typ, split="train", collector_args={}, repeat=1, image_size=[64,64]):
        super().__init__()

        image_collector = ImageCollector(data_path, typ, split, **collector_args)

        self.image_infos = []
        for i in range(repeat):
            self.image_infos += copy.deepcopy(image_collector.image_infos)

        transforms = [ttf.ToDtype(torch.uint8, scale=True),
                      ttf.Resize(size=image_size, antialias=True),
                      ttf.ToDtype(torch.float32, scale=True),
                      ttf.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])]
        self.transforms = ttf.Compose(transforms)

    def __len__(self):
        return len(self.image_infos)
    
    def __getitem__(self, index):
        info = self.image_infos[index]

        image = decode_image(info["path"], mode="RGB")
        #apply augmentations
        image = self.transforms(image)

        return {"image": image}

        





        
