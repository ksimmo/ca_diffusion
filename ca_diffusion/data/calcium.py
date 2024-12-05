import os

import numpy as np
import torch

import h5py

class CalciumDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=[64,64], num_crops_per_video=100, use_video_segments=True, augment_intensity=False):
        super().__init__()

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.num_crops_per_video = num_crops_per_video
        self.use_video_segments = use_video_segments
        self.augment_intensity = augment_intensity

        self.image_mode = len(crop_size)==2

        self.video_paths = []
        self.video_metadata = []
        self.descs = [] #h5 descriptors
        #load video information
        for f in os.listdir(os.path.join(root_dir, "data")):
            if f[0]=="0" or f=="test": #we do not want to load neurofinder videos for now
                continue

            p = os.path.join(self.root_dir, "data", f)

            try:
                h = h5py.File(p, "r")
                if not "images" in list(h.keys()):
                    h.close()
                    continue

                #read metadata
                metadata = {}
                metadata["shape"] = h["images"].shape
                metadata["max_intensity"] = np.amax(h["images"])
            except Exception as e:
                print(e)
                continue

            h.close()
            self.video_paths.append(p)
            self.video_metadata.append(metadata)
            self.descs.append(None)


    def __len__(self):
        return len(self.video_paths)*self.num_crops_per_video
    
    def check_h5(self, index):
        #TODO: convert dataset to webdataset
        #open h5 file if not opened -> not best practice
        if self.descs[index] == None:
            self.descs[index] = h5py.File(self.video_paths[index], "r")

    def augmentate(self, args):
        if np.random.uniform()>0.5:
            for i in range(len(args)):
                args[i] = np.flip(args[i], axis=1)
        if np.random.uniform()>0.5:
            for i in range(len(args)):
                args[i] = np.flip(args[i], axis=2)
        if np.random.uniform()>0.5:
            for i in range(len(args)):
                args[i] = np.rot90(args[i], axes=(1,2))

        #intensity augmentation
        if self.augment_intensity:
            scale_factor = np.random.uniform(0.7, 1.3)
            for i in range(len(args)):
                args[i] = np.round(args[i]*scale_factor) #round to nearest integer

            #TODO: maybe also add a constant offset

        for i in range(len(args)):
            if not self.image_mode:
                args[i] = torch.FloatTensor(args[i].copy()).unsqueeze(0) #add channel dim
            else:
                args[i] = torch.FloatTensor(args[i].copy()) #we can reuse the T as C axis
        return args
    
    def __getitem__(self, index):
        segment = index//len(self.video_paths)
        video_index = index%len(self.video_paths)
        self.check_h5(video_index)

        video_length = self.video_metadata[video_index]["shape"][0]
        video_size = self.video_metadata[video_index]["shape"][1:]
        segment_length = video_length//self.num_crops_per_video
        if self.image_mode and segment_length>0:
            #load random frame from current segment
            frame_index = np.random.randint(segment*segment_length, (segment+1)*segment_length)
        elif not self.image_mode and segment_length>self.crop_size[0]:
            #load random video from current segment
            frame_index = np.random.randint(segment*segment_length, (segment+1)*segment_length-self.crop_size[0])
        else:
            #random sample a frame
            if self.image_mode:
                frame_index = np.random.randint(0, video_length)
            else:
                frame_index = np.random.randint(0, video_length-self.crop_size[0])

        #now randomly sample a region from the video
        if self.image_mode:
            starty = np.random.randint(0, video_size[0]-self.crop_size[0])
            startx = np.random.randint(0, video_size[1]-self.crop_size[1])
            video = self.descs[video_index]["images"][frame_index:frame_index+1, starty:starty+self.crop_size[0], startx:startx+self.crop_size[1]]
        else:
            starty = np.random.randint(0, video_size[0]-self.crop_size[1])
            startx = np.random.randint(0, video_size[1]-self.crop_size[2])
            video = self.descs[video_index]["images"][frame_index:frame_index+self.crop_size[0], starty:starty+self.crop_size[1], startx:startx+self.crop_size[2]]

        #augmentate
        video = self.augmentate([video])[0]

        #most of the videos have a maximum around 4000, some videos however have over 60k
        max_intensity = self.video_metadata[video_index]["max_intensity"]
        if max_intensity>40000:
            video = video/16.0 #scale such that maximum is also around 4000
        video = video/4000.0 #scale by max
        video = (video-0.5)/0.5 #map to [-1,1]

        return {"data": video}