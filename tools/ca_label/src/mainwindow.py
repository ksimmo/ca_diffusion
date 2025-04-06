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



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("src/mainwindow.ui", self)

        #self.main_thread = MainThread(None)
        #self.main_thread.start()


        #setup widgets
        self.ui_label_video.linkSliders(self.ui_slider_shiftx, self.ui_slider_shifty, self.ui_slider_zoom, control_sliders=True)
        self.ui_label_flow.linkSliders(self.ui_slider_shiftx, self.ui_slider_shifty, self.ui_slider_zoom)
        self.ui_label_flow.poke_color = np.array([0,0,0], dtype=np.uint8)

        self.ui_slider_t.setRange(0,0)
        self.ui_slider_shiftx.setEnabled(False)
        self.ui_slider_shifty.setEnabled(False)
        self.ui_slider_zoom.setEnabled(False)

        #register callbacks
        self.ui_button_vselect.clicked.connect(self.OnVideoButtonSelect)
        self.ui_button_vload.clicked.connect(self.OnVideoButtonLoad)
        self.ui_button_vunload.clicked.connect(self.OnVideoButtonUnload)

        self.ui_button_mselect.clicked.connect(self.OnPokeButtonSelect)
        self.ui_button_msave.clicked.connect(self.OnPokeButtonSave)
        self.ui_button_mload.clicked.connect(self.OnPokeButtonLoad)
        self.ui_button_mclear.clicked.connect(self.OnPokeButtonClear)
        self.ui_button_mclear_all.clicked.connect(self.OnPokeButtonClearAll)

        self.ui_button_mask_add.clicked.connect(self.OnPokeButtonAdd)
        self.ui_button_mask_delete.clicked.connect(self.OnPokeButtonDelete)

        self.ui_button_calc.clicked.connect(self.OnVideoCalculate)
        self.ui_button_prev_t.clicked.connect(self.OnPrevFrame)
        self.ui_button_next_t.clicked.connect(self.OnNextFrame)
        self.ui_button_set_t.clicked.connect(self.OnSetFrame)
        self.ui_slider_t.sliderReleased.connect(self.OnSliderT)

        self.ui_label_video.moved.connect(self.OnMouseMove)
        self.ui_label_flow.moved.connect(self.OnMouseMove)
        self.ui_label_video.clicked.connect(self.OnMouseClicked)
        #self.ui_label_video.clicked.connect(self.ui_label_flow.changePokes)
        self.ui_label_flow.clicked.connect(self.OnMouseClicked)
        #self.ui_label_flow.clicked.connect(self.ui_label_video.changePokes)

        #stuff for handling video
        self.video_metadata = None
        self.video = None
        self.current_frame_index = 0
        self.masks = []

        #current frame display
        self.pixmap = QPixmap()
        self.clickable = True
        self.default_zoom = 2.0
        self.zoom_factor = 1.0 #the current zoom factor
        self.inuse = False

    def updateCanvas(self):
        img = self.image.astype(np.uint8)

        for i in range(len(self.pokes)):
            img[int(self.pokes[i][1]), int(self.pokes[i][0])] = self.poke_color #image has y first and then x

        val = 1.0
        if self.slider_zoom is not None:
            val = self.slider_zoom.value()
        fac = int(self.default_zoom*val)

        if self.inuse:
            #ok select correct region
            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

            img = img[shifty:shifty+1024//fac, shiftx:shiftx+1024//fac]

        #resize image
        img = resize(img, (img.shape[0]*fac, img.shape[1]*fac), order=0, preserve_range=True, anti_aliasing=False)

        #draw arrow
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        col = "#{:02x}{:02x}{:02x}".format(*self.poke_color)
        for i in range(len(self.pokes)):
            startx = int(self.pokes[i][0]*fac)
            starty = int(self.pokes[i][1]*fac)
            endx = int(self.pokes[i][2]*fac)
            endy = int(self.pokes[i][3]*fac)

            draw.line(((startx, starty), (endx, endy)), width=self.arrow_lw, fill=col)

            v = np.array([endx-startx, endy-starty])
            vlength = np.sqrt(np.sum(v**2))
            if vlength==0:
                continue

            arrow_head_height = min(vlength*0.2, 4) #arrow head height is at max 4 pixels
            arrow_head_intersect = -v/vlength*arrow_head_height+np.array([endx,endy])

            #find perpendicular vector
            normal = np.array([-v[1], v[0]])
            normal_length = np.sqrt(np.sum(normal**2))

            p1 = (arrow_head_intersect+normal/normal_length*arrow_head_height).astype(int)
            p2 = (arrow_head_intersect-normal/normal_length*arrow_head_height).astype(int)
            
            draw.polygon([(p1[0],p1[1]),(p2[0], p2[1]), (endx, endy)], fill=col) #draw arrow head

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
        self.pixmap = self.pixmap.fromImage(img)
        QLabel.setPixmap(self, self.pixmap)

    
    def mouseMoveEvent(self, ev: QMouseEvent):
        if self.inuse:
            #convert positions to pixels first
            val = 1.0
            if self.slider_zoom is not None:
                val = self.slider_zoom.value()
            fac = int(self.default_zoom*val)

            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

            pos = ev.position()
            x = int(pos.x()/fac)+shiftx
            y = int(pos.y()/fac)+shifty
            self.moved.emit(x, y) #convert to shifted coordinates

    def mousePressEvent(self, ev: QMouseEvent):
        if self.inuse:
            #convert positions to pixels first
            val = 1.0
            if self.slider_zoom is not None:
                val = self.slider_zoom.value()
            fac = int(self.default_zoom*val)

            shiftx = 0
            if self.slider_shiftx is not None:
                shiftx = int(self.slider_shiftx.value())

            shifty = 0
            if self.slider_shifty is not None:
                shifty = int(self.slider_shifty.value())

            pos = ev.position()
            x = int(pos.x()/fac)+shiftx
            y = int(pos.y()/fac)+shifty

            action = ev.button() #left click=1, right click=2
            if action==Qt.MouseButton.LeftButton: #set annotation
                action = 1
            elif action==Qt.MouseButton.RightButton: #remove annotation
                action = 0
            else:
                action = 0

            #check if we are inside the image
            if self.clickable and x<self.image.shape[1] and y<self.image.shape[0]:
                self.clicked.emit(x, y, action)

    def OnSliderZoom(self):
        val = 1.0
        if self.slider_zoom is not None:
            val = self.slider_zoom.value()
        fac = int(self.default_zoom*val)

        shape = np.array(self.image.shape[:2])*fac
        leftover = shape-1024

        #we only allow shifting of original pixels
        leftover = leftover//fac

        #set offset slider range
        if self.slider_shiftx is not None:
            if self.control_sliders and leftover[1]>0:
                shiftx = self.slider_shiftx.value()
                self.slider_shiftx.blockSignals(True)
                self.slider_shiftx.setRange(0,int(leftover[1]))
                #TODO: adapt shift slider to new zoom
                self.slider_shiftx.setValue(0)
                self.slider_shiftx.blockSignals(False)
                self.slider_shiftx.setEnabled(True)
            elif self.control_sliders and leftover[1]<=0:
                self.slider_shiftx.setEnabled(False)
        if self.slider_shifty is not None:
            if self.control_sliders and leftover[0]>0:
                shifty = self.slider_shifty.value()
                self.slider_shifty.blockSignals(True)
                self.slider_shifty.setRange(0,int(leftover[0]))
                #TODO: adapt shift slider to new zoom
                self.slider_shifty.setValue(0) #sliders are not updated! why???
                self.slider_shifty.blockSignals(False)
                self.slider_shifty.setEnabled(True)
            elif self.control_sliders and leftover[0]<=0:
                self.slider_shifty.setEnabled(False)

        self.updateCanvas()


    #override
    def closeEvent(self, event):
        #close stuff here
        #####
        QtWidgets.QMainWindow.closeEvent(self, event)

    #video loading
    def OnVideoButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "MP4 (*.mp4);;GIF (*.gif);;All files (*)")
        self.ui_edit_vpath.setText(fname[0])

    def OnVideoButtonLoad(self):
        self.OnVideoButtonUnload() #make sure everything is unloaded before
        path = self.ui_edit_vpath.text()

        if os.path.exists(path):
            try:
                self.log("Loading video from {} ...".format(path))
                #vr = decord.VideoReader(path)
                #num_frames = len(vr)
                #fps = vr.get_avg_fps()
                vr = VideoReader(path, "video")
                self.video_metadata = deepcopy(vr.get_metadata()["video"])

                frames = []
                for frame in vr:
                    frames.append(frame['data'])
                self.video = torch.stack(frames)
                self.video_metadata["num_frames"] = self.video.size(0)
                self.video_metadata["fps"] = self.video_metadata["fps"][0] 
                print(self.video_metadata)
                #vd = VideoDecoder(path)
                #self.video_metadata = deepcopy(vd.metadata)
                self.log("Video info: num_frames={} | fps={:.1f} H={} W={}".format(self.video_metadata["num_frames"], self.video_metadata["fps"], 
                                                                                   self.video.size(-2), self.video.size(-1)))
                del vr

                #add pokes
                for i in range(self.video_metadata["num_frames"]-1): #last frame does not have a flow to pick pokes from
                    self.pokes.append([]) #add an empty set pokes per frame

                self.ui_button_set_t.setEnabled(True)
                self.ui_button_next_t.setEnabled(True)
                self.ui_slider_t.setRange(0,self.video_metadata["num_frames"]-2) #last frame has no poke and indices start from 0
                #TODO: set slider t to 0
                self.ui_slider_t.setEnabled(True)
                self.ui_slider_zoom.setEnabled(True)
                self.SetCurrentFrame()

            except Exception as e:
                self.log("Cannot load video!", 1)
                print(e)

                self.video = None
                self.video_metadata = None
        else:
            self.log("Video {} does not exists!".format(path), 1)

    def OnVideoButtonUnload(self):
        self.video_metadata = None
        self.video = None
        self.current_frame_index = 0
        self.flow = None
        self.pokes = []

        #reset widgets

        self.ui_edit_t.blockSignals(True)
        self.ui_edit_t.setText("0")
        self.ui_edit_t.blockSignals(False)
        self.ui_button_set_t.setEnabled(False)
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
        #disabling is done via pixelmap

        self.ui_slider_shifty.blockSignals(True)
        self.ui_slider_shifty.setValue(0)
        self.ui_slider_shifty.setRange(0,0)
        self.ui_slider_shifty.blockSignals(False)
        #disabling is done via pixelmap

        self.ui_slider_zoom.blockSignals(True)
        self.ui_slider_zoom.setValue(1)
        self.ui_slider_zoom.blockSignals(False)
        self.ui_slider_zoom.setEnabled(False)

        self.ui_label_video.empty()
        self.ui_label_flow.empty()

        self.ui_combo_poke.clear()
        self.ui_edit_poke1.setText("")
        self.ui_edit_poke2.setText("")

        self.ui_label_flowmag.setText("0")
        self.ui_label_pos.setText("[PosX,PosY]")

        self.log("Current video unloaded!")


    #poke loading/saving
    def OnPokeButtonSelect(self):
        fname = QFileDialog.getOpenFileName(self, "Open file", "", "JSON (*.json);;All files (*)")
        self.ui_edit_ppath.setText(fname[0])

    def OnPokeButtonSave(self):
        path = self.ui_edit_ppath.text()

        if not path.endswith(".json"):
            self.log("File for saving pokes needs to be a JSON (.json) file!", 1)
            return

        #write pokes to file
        try:
            f = open(path, "w")
            pokes = []
            for i in range(len(self.pokes)):
                pokes.append([])
                for j in range(len(self.pokes[i])):
                    pokes[i].append(self.pokes[i][j].tolist())
            json.dump(pokes, f)
            f.close()
            self.log("Successfully saved pokes to {}".format(path))
        except Exception as e:
            self.log("Failed saving pokes to {}".format(path), 1)
            print(e)

    def OnPokeButtonLoad(self):
        path = self.ui_edit_ppath.text()

        if not path.endswith(".json"):
            self.log("File for loading pokes needs to be a JSON (.json) file!", 1)
            return

        #load pokes from file
        try:
            f = open(path, "r")
            pokes = json.load(f)
            self.pokes = []
            for i in range(len(pokes)):
                self.pokes.append([])
                for j in range(len(pokes[i])):
                    self.pokes[i].append(np.array(pokes[i][j]))
                    if i==self.current_frame_index:
                        self.ui_combo_poke.addItem("[{},{},{},{}]".format(pokes[i][j][0],pokes[i][j][1],pokes[i][j][2],pokes[i][j][3]))
            f.close()
            self.log("Successfully loaded pokes from {}".format(path))
        except Exception as e:
            self.log("Failed loading pokes from {}".format(path), 1)
            print(e)
            
            #reset pokes
            for i in range(self.video_metadata["num_frames"]-1):
                self.pokes.append([]) #add an empty set pokes per frame
        
        if len(self.pokes)<self.video_metadata["num_frames"]-1:
            missing = self.video_metadata["num_frames"]-1-len(self.pokes)
            for i in range(missing):
                self.pokes.append([])
            self.log("Loaded pokes are too short! Appending empty poke sets.", 1)
        elif len(self.pokes)>self.video_metadata["num_frames"]-1:
            missing = len(self.pokes)-self.video_metadata["num_frames"]-1
            for i in reversed(range(missing)):
                del self.pokes[-1]
            self.log("Loaded pokes are too long! Removing additional poke sets.", 1)

        self.ui_label_video.updatePokes(self.pokes[self.current_frame_index])
        self.ui_label_flow.updatePokes(self.pokes[self.current_frame_index])

    def OnPokeButtonClear(self):
        self.pokes[self.current_frame_index] = []
        self.ui_label_video.updatePokes()
        self.ui_label_flow.updatePokes()
        self.ui_combo_poke.clear()
        self.log("Clearing current frame pokes ...")
    
    def OnPokeButtonClearAll(self):
        for i in range(len(self.pokes)):
            self.pokes[i] = []
        self.ui_label_video.updatePokes()
        self.ui_label_flow.updatePokes()
        self.ui_combo_poke.clear()
        self.log("Clearing all pokes ...")

    ###########################
    def OnVideoCalculate(self):
        if self.video is None:
            return
        #get device
        device = self.ui_combo_device.currentText()
        device = torch.device(device)

        rel_flow = self.ui_checkbox_rel_flow.isChecked()
        text = self.ui_edit_chunk_size.text()
        chunk_size = int(text)

        self.setEnabled(False)
        start_time = time.time()
        self.raft_model = self.raft_model.to(device)
        padder = InputPadder(self.video.size(), mode="sintel")

        flows = []
        if rel_flow:
            start_frame = self.video[0:1].to(device)
            start_frame = (start_frame.float()/255.0-0.5)/0.5 #map to [-1,1]
            start_frame = padder.pad(start_frame)[0]
        chunk_size = self.video.size(0) if chunk_size<=0 else chunk_size #we are measuring in flow frames!
        num_chunks = int(np.ceil(self.video.size(0)/chunk_size))
        for i in range(num_chunks):
            start = i*chunk_size
            end = min(self.video.size(0), (i+1)*chunk_size+1)
            if end-start==1: #for last frame no flow exists anyways
                break
            clip = self.video[start:end].to(device)
            clip = (clip.float()/255.0-0.5)/0.5 #map to [-1,1]
            clip = padder.pad(clip)[0]
            if not rel_flow:
                flow = self.raft_model(clip[0:-1], clip[1:])[-1]
            else:
                flow = self.raft_model(start_frame.repeat(clip.size(0)-1,1,1,1), clip)[-1]
            flow = padder.unpad(flow)
            flows.append(flow.cpu())
        self.flow = torch.cat(flows, dim=0)
        if rel_flow:
            self.flow = self.flow[1:] #remove first flow frame as it will be just 0
            del start_frame
        del clip

        end_time = time.time()
        self.setEnabled(True)

        self.log("Flow calculated (time taken: {:.3f}s)!".format(end_time-start_time))

        self.SetCurrentFrame()

    def OnPrevFrame(self):
        if self.current_frame_index-1>=0:
            next_frame = self.current_frame_index-1
        else:
            next_frame = self.video.size(0)-1 #start again from last frame
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame() #update slider

    def OnNextFrame(self):
        if self.current_frame_index+1<self.video.size(0)-1:
            next_frame = self.current_frame_index+1
        else:
            next_frame = 0 #start again from first frame
        #get current postion
        self.ui_slider_t.blockSignals(True)
        self.ui_slider_t.setValue(next_frame)
        self.ui_slider_t.blockSignals(False)

        self.current_frame_index = next_frame
        self.SetCurrentFrame() #update slider

    def OnSetFrame(self):
        path = self.ui_edit_t.text()
        if not path.isnumeric():
            self.log("Frame index is not numeric!", 1)
            return
        
        index = int(path)
        if index<0 or index>=self.video.size(0)-1:
            self.log("Cannot set frame due to wrong length!",1)
            return
        
        if index!=self.current_frame_index:
            self.ui_slider_t.blockSignals(True)
            self.ui_slider_t.setValue(index)
            self.ui_slider_t.blockSignals(False)

            self.current_frame_index = index
            self.SetCurrentFrame() #update slider

    def OnSliderT(self):
        val = self.ui_slider_t.value()

        self.current_frame_index = val
        self.SetCurrentFrame()

    def SetCurrentFrame(self):
        if self.video is not None:
            #get current index
            frame = self.video[self.current_frame_index]
            self.ui_label_video.setImage(torch.movedim(frame, 0, 2).numpy(), self.pokes[self.current_frame_index])

            if self.flow is not None:
                flow = self.flow[self.current_frame_index]
                flow = flow_to_image(flow.unsqueeze(0).squeeze(0))
                self.ui_label_flow.setImage(torch.movedim(flow, 0, 2).numpy(), self.pokes[self.current_frame_index])
                mag = torch.sqrt(torch.sum(torch.pow(flow, 2),dim=1)).numpy()

            #add pokes to combobox
            self.ui_edit_poke1.setText("")
            self.ui_edit_poke2.setText("")
            self.ui_combo_poke.clear()
            for i in range(len(self.pokes[self.current_frame_index])):
                p = self.pokes[self.current_frame_index][i]
                self.ui_combo_poke.addItem("[{},{},{},{}]".format(p[0],p[1],p[2],p[3]))


    def OnMouseMove(self, x: int, y: int):
        self.ui_label_pos.setText("[{},{}]".format(x,y))
        if self.flow is not None:
            if x>=0 and x<self.flow.size(-1) and y>=0 and y<self.flow.size(-2):
                poke = self.flow[self.current_frame_index,:,y,x].numpy()
                pokemag = np.sqrt(np.sum(poke**2))
                self.ui_label_flowmag.setText("{}".format(int(pokemag)))

    def OnMouseClicked(self, x: int, y: int, action: int):
        found = False
        found_index = -1
        for i in range(len(self.pokes[self.current_frame_index])):
            if self.pokes[self.current_frame_index][i][0]==x and self.pokes[self.current_frame_index][i][1]==y:
                found = True
                found_index = i
                break

        if not found and action==1:
            self.log("Adding poke at x={} y={}".format(x, y))
            if self.flow is None:
                self.pokes[self.current_frame_index].append(np.array([x,y,0,0], dtype=int))
            else:
                flowx = int(self.flow[self.current_frame_index,0,y,x].item())
                flowy = int(self.flow[self.current_frame_index,1,y,x].item())
                self.pokes[self.current_frame_index].append(np.array([x,y,x+flowx,y+flowy], dtype=int))
            pokes = self.pokes[self.current_frame_index][-1]
            self.ui_combo_poke.addItem("[{},{},{},{}]".format(pokes[0], pokes[1], pokes[2], pokes[3]))
        elif found and action==0:
            del self.pokes[self.current_frame_index][found_index]
            self.ui_combo_poke.removeItem(found_index)
            self.log("Removing poke at x={} y={}".format(x, y))

        self.ui_label_video.updatePokes(self.pokes[self.current_frame_index])
        self.ui_label_flow.updatePokes(self.pokes[self.current_frame_index])

    def OnPokeButtonAdd(self):
        text1 = self.ui_edit_poke1.text()
        text2 = self.ui_edit_poke2.text()

        if not text1.isnumeric():
            self.log("Start position X is not numeric!", 1)
            return
        if not text2.isnumeric():
            self.log("Start position Y is not numeric!", 1)
            return
        
        pos1 = int(text1)
        pos2 = int(text2)

        if pos1<0 and pos1>self.video.size(-1):
            self.log("Start position X is out of bound!", 1)
            return
        if pos2<0 and pos2>self.video.size(-2):
            self.log("Start position Y is out of bound!", 1)
            return
        
        self.ui_label_video.pokes.append(np.array([pos1,pos2], dtype=int))
        self.ui_label_video.updateCanvas()
        self.ui_label_flow.pokes.append(np.array([pos1,pos2], dtype=int))
        self.ui_label_flow.updateCanvas()
        self.OnMouseClicked(pos1,pos2,1)   

    def OnPokeButtonDelete(self):
        if self.ui_combo_poke.count()>0:
            index = self.ui_combo_poke.currentIndex()
            text = self.ui_combo_poke.currentText()[1:-1] #remove []

            pos1 = text.find(",")
            x = int(text[:pos1])
            pos2 = text.find(",", pos1+1)
            y = int(text[pos1+1:pos2])
            
            self.OnMouseClicked(x,y,0) 