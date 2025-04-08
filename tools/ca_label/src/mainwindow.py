import sys
import os
import time

from copy import deepcopy
import json

from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6 import uic

import numpy as np
from skimage.transform import resize

from src.ca_worker import CalciumWorker



class MainWindow(QtWidgets.QMainWindow):
    worker_load = pyqtSignal(str, int)
    worker_unload = pyqtSignal()
    worker_pre_process = pyqtSignal()
    worker_process_save = pyqtSignal(str)
    worker_process_load = pyqtSignal(str)
    request_frame = pyqtSignal(str, int)
    measure_mask = pyqtSignal(np.ndarray, int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("src/mainwindow.ui", self)

        #setup widgets
        self.ui_slider_t.setRange(0,0)
        self.ui_slider_t.setEnabled(False)
        self.ui_slider_shiftx.setEnabled(False)
        self.ui_slider_shifty.setEnabled(False)
        self.ui_slider_zoom.setEnabled(False)
        self.ui_button_measure_run.setEnabled(False)
        self.ui_button_measure_clear.setEnabled(False)

        self.ui_combo_mode.addItem("Frame")
        self.ui_combo_mode.addItem("Mean")
        self.ui_combo_mode.addItem("Median")
        self.ui_combo_mode.addItem("Std")
        self.ui_combo_mode.addItem("Max.")
        self.ui_combo_mode.addItem("Max. Correlation")

        self.ui_combo_masks.clear()
        self.ui_combo_masks.addItem("None")

        #start ca worker
        self.ca_worker = CalciumWorker()
        self.ca_thread = QThread()
        self.ca_worker.moveToThread(self.ca_thread)
        self.ca_thread.start()


        #register callbacks
        self.ui_button_vselect.clicked.connect(self.OnVideoButtonSelect)
        self.ui_button_vload.clicked.connect(self.OnVideoButtonLoad)
        self.worker_load.connect(self.ca_worker.load_video)
        self.ca_worker.video_loaded.connect(self.OnVideoLoaded)
        self.ui_button_vunload.clicked.connect(self.OnVideoButtonUnload)
        self.worker_unload.connect(self.ca_worker.unload_video)

        self.ui_button_mselect.clicked.connect(self.OnMaskButtonSelect)
        self.ui_button_msave.clicked.connect(self.OnMaskButtonSave)
        self.ui_button_mload.clicked.connect(self.OnMaskButtonLoad)

        self.ui_combo_masks.currentIndexChanged.connect(self.OnMaskSelect)
        self.ui_button_mask_add.clicked.connect(self.OnMaskButtonAdd)
        self.ui_button_mask_delete.clicked.connect(self.OnMaskButtonDelete)
        self.ui_button_mask_delete_all.clicked.connect(self.OnMaskButtonDeleteAll)
        self.ui_button_mask_focus.clicked.connect(self.OnMaskButtonFocus)

        self.ui_button_pre_process.clicked.connect(self.OnPreProcessData)
        self.worker_pre_process.connect(self.ca_worker.pre_process_data)
        self.ca_worker.pre_process_finished.connect(self.OnPreProcessFinished)
        self.ca_worker.progress_update.connect(self.OnProgressUpdate)
        self.ui_button_psave.clicked.connect(self.OnProcessButtonSave)
        self.worker_process_save.connect(self.ca_worker.save_processed)
        self.ui_button_pload.clicked.connect(self.OnProcessButtonLoad)
        self.worker_process_load.connect(self.ca_worker.load_processed)

        self.ui_combo_mode.currentIndexChanged.connect(self.OnModeSelect)
        self.ui_slider_mask_alpha.sliderReleased.connect(self.updateView)
        self.ui_button_prev_t.clicked.connect(self.OnPrevFrame)
        self.ui_button_next_t.clicked.connect(self.OnNextFrame)
        self.ui_button_set_t.clicked.connect(self.OnSetFrame)
        self.ui_slider_t.sliderReleased.connect(self.OnSliderT)
        self.request_frame.connect(self.ca_worker.request_frame)
        self.ca_worker.transfer_frame.connect(self.OnReceiveFrame)
        self.ui_button_int_clip.clicked.connect(self.OnIntClip)

        self.ui_slider_shiftx.sliderReleased.connect(self.updateView)
        self.ui_slider_shifty.sliderReleased.connect(self.updateView)
        self.ui_slider_zoom.sliderReleased.connect(self.OnSliderZoom)

        self.ui_label_view.moved.connect(self.OnViewMouseMove)
        self.ui_label_view.clicked.connect(self.OnViewMouseClicked)


        self.ui_button_measure_run.clicked.connect(self.OnMeasureButtonRun)
        self.measure_mask.connect(self.ca_worker.request_track)
        self.ui_button_measure_clear.clicked.connect(self.OnMeasureButtonClear)


        #stuff for handling video
        self.is_video_loaded = False
        self.video_metadata = {}
        self.current_frame_index = 0
        self.masks = []
        self.current_mask_index = -1 #no mask selected

        #main display helpers
        self.default_zoom = 2.0 #keep this constant for now!
        self.zoom_factor = 1.0 #the current zoom factor
        self.inuse = False
        self.pixmap_view = QPixmap()
        self.image_view = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom)), dtype=np.uint8)
        self.image_view_contrast = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom)), dtype=np.uint8) #perform contrast calculations on here!

    #override
    def closeEvent(self, event):
        #close stuff here
        #####
        QtWidgets.QMainWindow.closeEvent(self, event)

    #video loading
    def OnVideoButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "All files (*);;Tiff (*.tiff);;H5 (*.h5)")
        self.ui_edit_vpath.setText(fname[0])

    def OnVideoButtonLoad(self):
        self.OnVideoButtonUnload() #make sure everything is unloaded before
        path = self.ui_edit_vpath.text()

        max_frames = self.ui_edit_max_frames.text()
        if len(max_frames)>0:
            if not max_frames.isnumeric():
                print("Number of frames is not numeric!")
                return
            max_frames = int(max_frames)
            if max_frames<0:
                max_frames = -1
            elif max_frames==0:
                print("You cannot load 0 frames!")
                return
        else:
            max_frames = -1

        self.worker_load.emit(path, max_frames)

    def OnVideoButtonUnload(self):
        self.worker_unload.emit()

        self.is_video_loaded = False
        self.video_metadata = {}
        self.current_frame_index = 0
        self.masks = []
        self.current_mask_index = -1
        self.zoom_factor = 1.0
        self.inuse = False

        #reset and disable widgets
        self.ui_edit_t.blockSignals(True)
        self.ui_edit_t.setText("0")
        self.ui_edit_t.blockSignals(False)
        self.ui_button_set_t.setEnabled(False)
        self.ui_button_prev_t.setEnabled(False)
        self.ui_button_next_t.setEnabled(False)

        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(0)
        self.ui_slider_t.setRange(0,0)
        self.ui_slider_t.blockSignals(False)
        self.ui_slider_t.setEnabled(False)

        self.ui_slider_shiftx.blockSignals(True)
        self.ui_slider_shiftx.setValue(0)
        self.ui_slider_shiftx.setRange(0,0)
        self.ui_slider_shiftx.blockSignals(False)
        self.ui_slider_shiftx.setEnabled(False)

        self.ui_slider_shifty.blockSignals(True)
        self.ui_slider_shifty.setValue(0)
        self.ui_slider_shifty.setRange(0,0)
        self.ui_slider_shifty.blockSignals(False)
        self.ui_slider_shifty.setEnabled(False)

        self.ui_slider_zoom.blockSignals(True)
        self.ui_slider_zoom.setValue(1)
        self.ui_slider_zoom.blockSignals(False)
        self.ui_slider_zoom.setEnabled(False)

        self.ui_combo_mode.blockSignals(True)
        self.ui_combo_mode.setCurrentIndex(0) #default mode is frame
        self.ui_combo_mode.blockSignals(False)
        self.ui_combo_mode.setEnabled(False)

        self.ui_label_view.clear()
        self.image_view = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom)), dtype=np.uint8)
        self.image_view_contrast = np.zeros((int(1024/self.default_zoom), int(1024/self.default_zoom)), dtype=np.uint8)
        self.ui_label_signal.clear()

        self.ui_button_pre_process.setEnabled(True)

        self.ui_label_value2.setText("0")
        self.ui_label_pos.setText("[PosX,PosY]")

        self.ui_edit_int_low.setText("")
        self.ui_edit_int_high.setText("")
        self.ui_button_int_clip.setEnabled(False)

        self.ui_button_measure_run.setEnabled(False)
        self.ui_button_measure_clear.setEnabled(False)

        print("Current Video Unloaded")

    def OnVideoLoaded(self, num_frames: int, frame0: np.ndarray):
        """
        Our video worker has loaded the video
        """
        self.video_metadata["num_frames"] = num_frames
        self.is_video_loaded = True
        self.image_view = frame0
        self.OnIntClip()

        #re-enable sliders
        self.ui_slider_shiftx.setEnabled(True)
        self.ui_slider_shifty.setEnabled(True)
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setRange(0,num_frames-1)
        self.ui_slider_t.blockSignals(False)
        self.ui_slider_t.setEnabled(True)
        self.ui_slider_zoom.setEnabled(True)

        self.ui_button_set_t.setEnabled(True)
        self.ui_button_prev_t.setEnabled(True)
        self.ui_button_next_t.setEnabled(True)

        self.ui_button_int_clip.setEnabled(True)

        self.ui_button_measure_run.setEnabled(True)
        self.ui_button_measure_clear.setEnabled(True)

        print("Successfully loaded video!", num_frames)

    #################
    #mask loading/saving
    def OnMaskButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "JSON (*.json);;All files (*)")
        self.ui_edit_mpath.setText(fname[0])

    def OnMaskButtonSave(self):
        path = self.ui_edit_mpath.text()

        if not path.endswith(".json"):
            print("File for saving masks needs to be a JSON (.json) file!")
            return

        #write mask to file
        try:
            f = open(path, "w")
            masks = []
            for m in self.masks:
                masks.append({"id": m["id"], "coordinates": m.tolist()})
            json.dump(masks, f)
            f.close()
            print("Successfully saved masks to {}".format(path))
        except Exception as e:
            print("Failed saving masks to {}".format(path))
            print(e)

    def OnMaskButtonLoad(self):
        path = self.ui_edit_mpath.text()

        if not path.endswith(".json"):
            print("File for loading masks needs to be a JSON (.json) file!")
            return

        #load masks from file
        try:
            f = open(path, "r")
            self.masks = json.load(f)
            for m in self.masks:
                m["coordinates"] = np.array(m["coordinates"])
            f.close()

            self.ui_combo_masks.clear()
            self.ui_combo_masks.addItem("None")
            for m in self.masks:
                self.ui_combo_masks.addItem("ID: {}".format(m["id"]))

            print("Successfully loaded masks from {}".format(path))


        except Exception as e:
            print("Failed loading masks from {}".format(path))
            print(e)
            
            #reset masks
            self.masks = []

        self.updateView()

    def OnMaskSelect(self):
        if self.ui_combo_masks.count()>1: #1 is just None mask
            index = self.ui_combo_masks.currentIndex()
            self.current_mask_index = index-1

            self.updateView()

    def OnMaskButtonAdd(self):
        ids = []
        for m in self.masks:
            ids.append(m["id"])

        #add mask with non existing id
        counter = 0
        while "{}".format(counter) in ids:
            counter +=1
        
        newmask = {"id": "{}".format(counter), "coordinates": np.zeros((0,2), dtype=int)}
        self.masks.append(newmask)
        self.ui_combo_masks.blockSignals(True)
        self.ui_combo_masks.addItem("ID: {}".format(newmask["id"]))
        self.ui_combo_masks.setCurrentIndex(len(self.masks))
        self.ui_combo_masks.blockSignals(False)

        self.current_mask_index = len(self.masks)-1

        self.updateView()

    def OnMaskButtonDelete(self):
        if self.ui_combo_masks.count()>1: #first entry is just None mask
            index = self.ui_combo_masks.currentIndex()
            if index>0: #cannot remove None mask
                self.ui_combo_masks.removeItem(index)
                del self.masks[index-1]

                #chose no current active mask
                self.ui_combo_masks.blockSignals(True)
                self.ui_combo_masks.setCurrentIndex(0)
                self.ui_combo_masks.blockSignals(False)
                self.current_mask_index = -1

                self.updateView()

    def OnMaskButtonDeleteAll(self):
        print("Deleting all masks!")
        self.masks = []
        self.ui_combo_masks.clear()
        self.ui_combo_masks.addItem("None")
        self.current_mask_index = -1

        self.updateView()

    def OnMaskButtonFocus(self):
        if self.current_mask_index>=0:
            coordinates = self.masks[self.current_mask_index]["coordinates"]
            mins = np.amin(coordinates, axis=0)
            maxs = np.amax(coordinates, axis=0)+1

            center = (mins+maxs)*0.5

            #TODO: adjust slidersx and y such that current mask pops up in current view
            val = self.ui_slider_zoom.value()
            fac = int(self.default_zoom*val)

            self.ui_slider_shiftx.blockSignals(True)
            self.ui_slider_shiftx.setValue(0) #...
            self.ui_slider_shiftx.blockSignals(False)

            self.ui_slider_shifty.blockSignals(True)
            self.ui_slider_shifty.setValue(0) #...
            self.ui_slider_shifty.blockSignals(False)

            self.updateView()

    #################
    def updateView(self):
        img = np.expand_dims(np.copy(self.image_view_contrast), -1)
        img = np.repeat(img, 3, axis=-1)

        #normalize intensity such that it fits [0,255]
        amax = np.amax(img)
        img = img/amax*255.0

        #draw masks
        img2 = np.copy(img)
        #directly draw onto image
        for i,m in enumerate(self.masks):
            if len(m["coordinates"])==0:
                continue
            if i==self.current_mask_index:
                img2[m["coordinates"][:,0], m["coordinates"][:,1]] = np.array([255.0,0,0])
            else:
                img2[m["coordinates"][:,0], m["coordinates"][:,1]] = np.array([0,255.0,0])

        alpha = self.ui_slider_mask_alpha.value()/100.0

        img = img2*alpha+(1.0-alpha)*img #blend masks in
        

        val = self.ui_slider_zoom.value()
        fac = int(self.default_zoom*val)

        #ok select correct region
        shiftx = int(self.ui_slider_shiftx.value())
        shifty = int(self.ui_slider_shifty.value())

        img = img[shifty:shifty+1024//fac, shiftx:shiftx+1024//fac]

        #resize image
        img = resize(img, (img.shape[0]*fac, img.shape[1]*fac), order=0, preserve_range=True, anti_aliasing=False)

        img = np.array(img)

        #pad image if necessary!!!
        if img.shape[0]<1024:
            missing = 1024-img.shape[0]
            img = np.concatenate([img, np.zeros_like(img[:missing])], axis=0)

        if img.shape[1]<1024:
            missing = 1024-img.shape[1]
            img = np.concatenate([img, np.zeros_like(img[:,:missing])], axis=1)

        #draw new image onto label
        img = QImage(img.astype(np.uint8), img.shape[1], img.shape[0], 3*img.shape[1], QImage.Format.Format_RGB888)
        self.pixmap_view = self.pixmap_view.fromImage(img)
        self.ui_label_view.setPixmap(self.pixmap_view)

    def OnSliderZoom(self):
        val = self.ui_slider_zoom.value()
        fac = int(self.default_zoom*val)

        shape = np.array(self.image_view.shape[:2])*fac
        leftover = shape-1024

        #we only allow shifting of original pixels
        leftover = leftover//fac

        #set offset slider range
        #sliderx
        if leftover[1]>0:
            #TODO: rembember old position before zoom
            shiftx = self.ui_slider_shiftx.value()
            self.ui_slider_shiftx.blockSignals(True)
            self.ui_slider_shiftx.setRange(0,int(leftover[1]))
            self.ui_slider_shiftx.setValue(0)
            self.ui_slider_shiftx.blockSignals(False)
            self.ui_slider_shiftx.setEnabled(True)
        elif leftover[1]<=0:
            self.ui_slider_shiftx.setEnabled(False)

        #slidery
        if leftover[0]>0:
            #TODO: rembember old position before zoom
            shifty = self.ui_slider_shifty.value()
            self.ui_slider_shifty.blockSignals(True)
            self.ui_slider_shifty.setRange(0,int(leftover[0]))
            self.ui_slider_shifty.setValue(0) #sliders are not updated! why???
            self.ui_slider_shifty.blockSignals(False)
            self.ui_slider_shifty.setEnabled(True)
        elif leftover[0]<=0:
            self.ui_slider_shifty.setEnabled(False)

        self.updateView()


    def OnPreProcessData(self):
        if self.is_video_loaded:
            self.ui_button_pre_process.setEnabled(False)
            self.worker_pre_process.emit()

    def OnPreProcessFinished(self):
        self.ui_button_pre_process.setEnabled(True)
        self.ui_combo_mode.setEnabled(True)

    def OnProgressUpdate(self, value: int):
        self.ui_progress.setValue(value)

    def OnProcessButtonSave(self):
        path = self.ui_edit_ppath.text()
        self.worker_process_save.emit(path)

    def OnProcessButtonLoad(self):
        path = self.ui_edit_ppath.text()
        self.worker_process_load.emit(path)

    def OnIntClip(self):
        low = self.ui_edit_int_low.text()
        if len(low)>0:
            if low.isnumeric():
                low = int(low)
            else:
                low = None
        else:
            low = None

        high = self.ui_edit_int_high.text()
        if len(high)>0:
            if high.isnumeric():
                high = int(high)
            else:
                high = None
        else:
            high = None

        self.image_view_contrast = np.copy(self.image_view)
        if high is not None:
            self.image_view_contrast[self.image_view_contrast>high] = high
        if low is not None:
            self.image_view_contrast -= low
            self.image_view_contrast[self.image_view_contrast<0] = 0

        self.updateView()

    def OnModeSelect(self):
        index = self.ui_combo_mode.currentIndex()

        if index==0:
            self.request_frame.emit("frame", self.current_frame_index)
        elif index==1:
            self.request_frame.emit("mean", -1)
        elif index==2:
            self.request_frame.emit("median", -1)
        elif index==3:
            self.request_frame.emit("std", -1)
        elif index==4:
            self.request_frame.emit("max", -1)
        elif index==5:
            self.request_frame.emit("max_corr", -1)

    def OnPrevFrame(self):
        if self.current_frame_index-1>=0:
            next_frame = self.current_frame_index-1
        else:
            next_frame = self.video_metadata["num_frames"]-1 #start again from last frame
            
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame()

    def OnNextFrame(self):
        if self.current_frame_index+1<self.video_metadata["num_frames"]:
            next_frame = self.current_frame_index+1
        else:
            next_frame = 0 #start again from first frame
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame()

    def OnSetFrame(self):
        path = self.ui_edit_t.text()
        if not path.isnumeric():
            print("Frame index is not numeric!")
            return
        
        index = int(path)
        if index<0 or index>=self.video_metadata["num_frames"]:
            print("Cannot set frame due to wrong length!")
            return
        
        if index!=self.current_frame_index:
            self.ui_slider_t.blockSignals(True)
            self.ui_slider_t.setValue(index)
            self.ui_slider_t.blockSignals(False)

            self.current_frame_index = index
            self.SetCurrentFrame()

    def OnSliderT(self):
        val = self.ui_slider_t.value()

        self.current_frame_index = val
        self.SetCurrentFrame()

    def SetCurrentFrame(self):
        if self.is_video_loaded:
            curmode = self.ui_combo_mode.currentIndex()
            if curmode==0:
                self.request_frame.emit("frame", self.current_frame_index)

    def OnReceiveFrame(self, frame):
        self.image_view = frame
        self.OnIntClip()

    def OnViewMouseMove(self, x: int, y: int):
        val = self.ui_slider_zoom.value()
        fac = int(self.default_zoom*val)
        shiftx = int(self.ui_slider_shiftx.value())
        shifty = int(self.ui_slider_shifty.value())

        x = int(x/fac)+shiftx
        y = int(y/fac)+shifty

        self.ui_label_pos.setText("[{},{}]".format(x,y))
        intensity = int(self.image_view[y,x])
        self.ui_label_value2.setText("{}".format(intensity))

    def OnViewMouseClicked(self, x: int, y: int, action: int):
        val = self.ui_slider_zoom.value()
        fac = int(self.default_zoom*val)
        shiftx = int(self.ui_slider_shiftx.value())
        shifty = int(self.ui_slider_shifty.value())

        x = int(x/fac)+shiftx
        y = int(y/fac)+shifty

        if x<self.image_view.shape[1] and y<self.image_view.shape[0]:
            if self.current_mask_index>=0:
                #check if pixel is in list
                array = np.array([[y,x]])

                is_in_list = np.all(array==self.masks[self.current_mask_index]["coordinates"], axis=1)
                print(is_in_list)

                if not np.any(is_in_list) and action==1:
                    self.masks[self.current_mask_index]["coordinates"] = np.concatenate([self.masks[self.current_mask_index]["coordinates"], array], axis=0)
                    self.updateView()
                elif np.any(is_in_list) and action==0:
                    index = np.where(is_in_list)[0]
                    self.masks[self.current_mask_index]["coordinates"] = np.delete(self.masks[self.current_mask_index]["coordinates"], index, axis=0)
                    self.updateView()

    ###################
    def OnMeasureButtonRun(self):
        if self.current_mask_index>=0:
            safety_margin = self.ui_slider_safety_margin.value()
            background_margin = self.ui_slider_background_margin.value()
            self.measure_mask.emit(self.masks[self.current_mask_index]["coordinates"], safety_margin, background_margin)

    def OnMeasureReceive(self):
        pass

    def OnMeasureButtonClear(self):
        self.ui_label_signal.clear()
        self.ui_label_histogram.clear()