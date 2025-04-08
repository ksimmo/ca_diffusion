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
    transfer_tracks = pyqtSignal(np.ndarray) #send tracks (raw, bg, neuron only)

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

    def pre_process_data(self):
        if len(self.data_calculated.keys())==0: #only pre-process if not done already
            self.progress_update.emit(0)
            self.data_calculated["max_intensity"] = np.amax(self.data)
            self.data_calculated["mean_intensity"] = np.mean(self.data)
            self.data_calculated["std_intensity"] = np.std(self.data)
            self.data_calculated["median_intensity"] = np.median(self.data)
            self.data_calculated["mad_intensity"] = np.median(self.data-self.data_calculated["median_intensity"])

            self.progress_update.emit(20)

            self.data_calculated["mean_image"] = np.mean(self.data, axis=0)
            self.progress_update.emit(60)
            self.data_calculated["std_image"] = np.std(self.data, axis=0)
            self.progress_update.emit(70)
            #self.data_calculated["median_image"] = np.median(self.data, axis=0)
            self.progress_update.emit(80)
            self.data_calculated["max_image"] = np.amax(self.data, axis=0)
            self.progress_update.emit(90)

            #correlation image?

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
        elif variant=="max_corr" and "max_corr_image" in self.data_calculated.keys():
            self.transfer_frame.emit(self.data_calculated["max_corr_image_image"])

    def request_track(self, coordinates, safety_margin=0, bg_margin=0):
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
        #TODO: take care of other masks in that region
        if safety_margin>0:
            safety_mask = np.copy(mask)
            safety_mask = binary_dilation(safety_mask, iterations=safety_margin)
            
        if bg_margin>0:
            bg_mask = np.copy(mask)
            bg_mask = binary_dilation(bg_mask, iterations=safety_margin+bg_margin)
            if safety_margin>0:
                bg_mask = bg_mask-safety_margin
            else:
                bg_mask = bg_mask-mask

        crop = self.data[:,mins[0]:maxs[0], mins[1]:maxs[1]]

        #get average signal and histograms
        raw_signal = np.sum(crop*np.expand_dims(mask, axis=0), axis=(1,2))/np.sum(mask)
        print(raw_signal.shape)

        self.transfer_tracks.emit(raw_signal)

