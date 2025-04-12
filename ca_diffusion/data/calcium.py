import os

import numpy as np
import torch

import h5py
import json

class CalciumDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, crop_size=[64,64], num_crops_per_video=100, use_video_segments=True, augment_intensity=False, load_annotations=False,
                        fixed_intensity_binning=False):
        super().__init__()

        self.root_dir = root_dir
        self.crop_size = crop_size
        self.num_crops_per_video = num_crops_per_video
        self.use_video_segments = use_video_segments
        self.augment_intensity = augment_intensity
        self.load_annotations = load_annotations
        self.fixed_intensity_binning = fixed_intensity_binning

        self.image_mode = len(crop_size)==2
        self.video_names = []
        self.video_metadata = []
        self.descs = [] #h5 descriptors
        #load video information
        for f in os.listdir(root_dir):
            if "test" in f: #do not load test videos
                continue

            try:
                h = h5py.File(os.path.join(self.root_dir, f, "data", "{}.h5".format(f)), "r")
                if not "images" in list(h.keys()):
                    h.close()
                    continue

                #read metadata
                metadata = {}
                metadata["shape"] = h["images"].shape
                metadata["max_intensity"] = float(h["images"].attrs["max_intensity"])
                metadata["mean_intensity"] = float(h["images"].attrs["mean_intensity"])
                metadata["std_intensity"] = float(h["images"].attrs["std_intensity"])
            except Exception as e:
                print(e)
                continue

            h.close()
            self.video_names.append(f)
            self.video_metadata.append(metadata)
            self.descs.append(None)


    def __len__(self):
        return len(self.video_names)*self.num_crops_per_video
    
    def check_h5(self, index):
        #TODO: convert dataset to webdataset
        #open h5 file if not opened -> not best practice
        if self.descs[index] == None:
            self.descs[index] = h5py.File(os.path.join(self.root_dir, self.video_names[index], "data", "{}.h5".format(self.video_names[index])), "r")

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
        segment = index//len(self.video_names)
        video_index = index%len(self.video_names)
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

        data = [video]
        if self.load_annotations:
            f = open(os.path.join(self.root_dir, self.video_names[video_index], "data", "regions.json"), "r")
            annotations = json.load(f)
            segmap = np.zeros(video_size)
            for a in annotations:
                coords = np.array(a["coordinates"])
                segmap[coords[:,0], coords[:,1]] = 1
            data.append(torch.FloatTensor(segmap).unsqueeze(0))

        #augmentate

        data = self.augmentate(data)
        video = data[0]

        #most of the videos have a maximum around 4000, some videos however have over 60k
        if self.fixed_intensity_binning:
            #make sure each video has the same intensity bin size
            max_intensity = self.video_metadata[video_index]["max_intensity"]
            if max_intensity>4000:
                #rescale video nad fix the bins
                video = video/max_intensity*4000.0
                video = np.floor(video)
            video = video/4000.0
        else:
            max_intensity = self.video_metadata[video_index]["max_intensity"]
            video = video/max_intensity
        video = (video-0.5)/0.5

        element = {"image": video}
        if len(data)==2:
            element["segmap"] = data[1]
        return element
    

class CalciumMean(CalciumDataset):
    def __init__(self, root_dir, crop_size=[64,64], num_crops_per_video=100, spatial_split=False, validation=False):
        assert len(crop_size)==2, "Only spatial crops are supported!"
        super().__init__(root_dir=root_dir, crop_size=crop_size, num_crops_per_video=num_crops_per_video)

        self.spatial_split = spatial_split #use top quarter for validation
        self.is_validation = validation

        #pre calculate average projections and segmentation maps
        self.segmaps = []
        self.mean_projections = []
        for i,v in enumerate(self.video_names):
            f = open(os.path.join(self.root_dir, v, "data", "regions.json"), "r")
            annotations = json.load(f)
            f.close()
            segmap = np.zeros(self.video_metadata[i]["shape"][1:])
            for a in annotations:
                coords = np.array(a["coordinates"])
                segmap[coords[:,0], coords[:,1]] = 1

            self.segmaps.append(segmap)

            h = h5py.File(os.path.join(self.root_dir, v, "data", "{}.h5".format(v)), "r")

            self.mean_projections.append(np.mean(h["images"][:], axis=(1,2)))
            h.close()


    def __getitem__(self, index):
        video_index = index%len(self.video_names)
        
        #sample region
        shape = self.video_metadata[video_index]["shape"][1:]
        if self.spatial_split:
            if self.is_validation: #top left quarter is for validation
                starty = np.random.randint(shape[0]//2-self.crop_size[0])
                startx = np.random.randint(shape[1]//2-self.crop_size[1])
            else:
                starty = np.random.randint(shape[0]-self.crop_size[0])
                if starty<shape[0]//2:
                    startx = np.random.randint(shape[1]//2, shape[1]-self.crop_size[1])
                else:
                    startx = np.random.randint(shape[1]-self.crop_size[1])
        else:
            starty = np.random.randint(shape[0]-self.crop_size[0])
            startx = np.random.randint(shape[1]-self.crop_size[1])

        crop = self.mean_projections[video_index][starty:starty+self.crop_size[0], startx:startx+self.crop_size[1]]
        seg = self.segmaps[video_index][starty:starty+self.crop_size[0], startx:startx+self.crop_size[1]]

        #augmentate
        data = [np.expand_dims(crop, axis=0), np.expand_dims(seg, axis=0)]
        data = self.augmentate(data)

        data[0] = (data[0]-self.video_metadata[video_index]["mean_intensity"])/self.video_metadata[video_index]["std_intensity"]
        
        return {"mean_proj": data[0], "segmap": data[1]}

    
