from PyQt6.QtCore import QObject, pyqtSignal

import numpy as np
import h5py
import tifffile

from scipy.ndimage import binary_dilation

class CalciumWorker(QObject):
    video_loaded =pyqtSignal(int, np.ndarray) #when we successfully loaded a video
    progress_update = pyqtSignal(int) #update the progress bar in main window
    pre_process_finished = pyqtSignal()
    transfer_frame = pyqtSignal(np.ndarray) #send a single frame for drawing
    transfer_measured = pyqtSignal(dict) #send tracks (raw, bg, neuron only)

    def __init__(self):
        super().__init__()
        self.data = None
        self.data_calculated = {}

    def load_video(self, path: str, max_frames: int):
        print("Trying to load video from {}".format(path))

        if path.endswith(".tiff"):
            try:
                self.data = tifffile.imread(path)
                if self.data.shape[0]>max_frames and max_frames>0:
                    self.data = self.data[:max_frames]
                self.video_loaded.emit(self.data.shape[0], self.data[0])
            except Exception as e:
                print("Cannot load video!")
                print(e)
        elif path.endswith(".h5"):
            try:
                f = h5py.File(path, "r")
                if f["images"].shape[0]>max_frames and max_frames>0:
                    self.data = f["images"][:max_frames] 
                else:
                    self.data = f["images"][:] #load data, for now we only support laoding full data into memory
                f.close()
                self.video_loaded.emit(self.data.shape[0], self.data[0])
            except Exception as e:
                print("Cannot load video!")
                print(e)

    def unload_video(self):
        self.data = None
        self.data_calculated = {}

    def pre_process_data(self, corr_neighbourhood=3):
        if len(self.data_calculated.keys())==0: #only pre-process if not done already
            self.progress_update.emit(0)
            self.data_calculated["max_intensity"] = np.amax(self.data)
            self.data_calculated["mean_intensity"] = np.mean(self.data)
            self.data_calculated["std_intensity"] = np.std(self.data)
            self.data_calculated["median_intensity"] = np.median(self.data)
            self.data_calculated["mad_intensity"] = np.median(self.data-self.data_calculated["median_intensity"])
            self.progress_update.emit(10)

            self.data_calculated["mean_image"] = np.mean(self.data, axis=0)
            self.progress_update.emit(20)
            self.data_calculated["std_image"] = np.std(self.data, axis=0)
            self.progress_update.emit(30)
            #self.data_calculated["median_image"] = np.median(self.data, axis=0)
            self.progress_update.emit(50)
            self.data_calculated["max_image"] = np.amax(self.data, axis=0)
            self.progress_update.emit(60)

            #correlation image (lazy version)
            if corr_neighbourhood>0:
                corr_image_avg = np.zeros_like(self.data_calculated["mean_image"])
                corr_image_max = np.zeros_like(self.data_calculated["mean_image"])
                diff = self.data-np.expand_dims(self.data_calculated["mean_image"], axis=0)
                for i in range(corr_neighbourhood, self.data.shape[1]-corr_neighbourhood):
                    for j in range(corr_neighbourhood, self.data.shape[2]-corr_neighbourhood):
                        crop = diff[:,i-corr_neighbourhood:i+corr_neighbourhood+1,j-corr_neighbourhood:j+corr_neighbourhood+1]
                        crop = crop.reshape(crop.shape[0],-1)
                        crop_std = self.data_calculated["std_image"][i-corr_neighbourhood:i+corr_neighbourhood+1,j-corr_neighbourhood:j+corr_neighbourhood+1]
                        crop_std = crop_std.flatten()

                        center_i = crop_std.shape[0]//2

                        #remove self-correlation
                        crop = np.delete(crop, center_i, axis=1)
                        crop_std = np.delete(crop_std, center_i, axis=0)

                        corr = np.sum(diff[:,i,j].reshape(-1,1)*crop, axis=0)/(self.data_calculated["std_image"][i,j]*crop_std)
                        corr_image_avg[i,j] = np.mean(corr)/self.data.shape[0]
                        corr_image_max[i,j] = np.amax(corr/self.data.shape[0])
                    self.progress_update.emit(int(i/self.data.shape[1]*40)+60)
                self.data_calculated["corr_image_avg"] = corr_image_avg+1.0 #move from [-1,1] to [0,2] to match normalization in main view
                self.data_calculated["corr_image_max"] = corr_image_max+1.0

            self.progress_update.emit(100)
            self.pre_process_finished.emit()

    def save_processed(self, path):
        if path.endswith(".npz"):
            np.savez(path, **self.data_calculated)
            print("Saved preprocessed data to {}!".format(path))
        else:
            print("Wrong file format!")

    def load_processed(self, path):
        try:
            self.data_calculated = dict(np.load(path))
            print("Loaded preprocessed data from {}!".format(path))
            self.pre_process_finished.emit()
        except Exception as e:
            print("Cannot load preprocessed data from {}!".format(path))


    def request_frame(self, variant="frame", index=None):
        if variant=="frame":
            if index>=0 and index<self.data.shape[0]:
                self.transfer_frame.emit(self.data[index])
        elif variant=="mean" and "mean_image" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["mean_image"])
        elif variant=="std" and "std_image" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["std_image"])
        elif variant=="max" and "max_image" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["max_image"])
        elif variant=="median" and "median_image" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["median_image"])
        elif variant=="corr_avg" and "corr_image_avg" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["corr_image_avg"])
        elif variant=="corr_max" and "corr_image_max" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["corr_image_max"])

    def request_measure(self, coordinates, mask_map, safety_margin=0, bg_margin=0):
        mins = np.amin(coordinates, axis=0)
        maxs = np.amax(coordinates, axis=0)+1
        #add margins to maximum crop size
        mins = mins-safety_margin-bg_margin
        maxs = maxs+safety_margin+bg_margin

        #check if we exceed data size
        mins = np.maximum(mins, np.zeros(2)).astype(int)
        maxs = np.minimum(maxs, self.data.shape[1:]).astype(int)

        mask = np.zeros((maxs-mins).astype(int))
        mask[coordinates[:,0]-mins[0], coordinates[:,1]-mins[1]] = 1.0

        #perform opening operations to enlarge masks to get safety and background mask if necessary
        safety_mask = None
        if safety_margin>0 and bg_margin>0: #we only need safety margin for bg
            safety_mask = np.copy(mask_map[mins[0]:maxs[0], mins[1]:maxs[1]]) #calculate safety margin around every mask
            safety_mask = binary_dilation(safety_mask, iterations=safety_margin).astype(float)
        
        bg_mask = None
        if bg_margin>0:
            bg_mask = np.copy(mask)
            bg_mask = binary_dilation(bg_mask, iterations=safety_margin+bg_margin).astype(float)
            if safety_mask is not None:
                bg_mask = bg_mask-safety_mask
            else:
                bg_mask = bg_mask-mask_map[mins[0]:maxs[0], mins[1]:maxs[1]]
            bg_mask[bg_mask<0] = 0.0

        crop = self.data[:,mins[0]:maxs[0], mins[1]:maxs[1]]

        #get average signal and histograms
        raw_signal = np.sum(crop*np.expand_dims(mask, axis=0), axis=(1,2))/np.sum(mask)

        tmask = np.repeat(np.expand_dims(mask, axis=0),crop.shape[0], axis=0)
        raw_data = crop[np.where(tmask>0)]
        amax = int(np.amax(self.data))
        h, ed = np.histogram(raw_data, bins=amax+1, range=(0,amax+1), density=True)

        temp = {"raw_signal": raw_signal, "raw_dist": h}
        if bg_mask is not None:
            bg_signal = np.sum(crop*np.expand_dims(bg_mask, axis=0), axis=(1,2))/np.sum(bg_mask)
            temp["bg_signal"] = bg_signal

            tmask = np.repeat(np.expand_dims(bg_mask, axis=0),crop.shape[0], axis=0)
            raw_data_bg = crop[np.where(tmask>0)]
            h2, ed = np.histogram(raw_data_bg, bins=amax+1, range=(0,amax+1), density=True)
            temp["bg_dist"] = h2

            #temp["fg_signal"] = raw_signal-0.7*bg_signal #maybe optimize for r

        self.transfer_measured.emit(temp)

