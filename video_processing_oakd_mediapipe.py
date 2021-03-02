import traceback
from pathlib import Path
import cv2
import sys
import argparse
import os

import depthai as dai
import numpy as np
from collections import namedtuple

import time

# OAK-D hand tracking using mediapipe converted from tesnsorflow lite models
# https://github.com/geaxgx/depthai_hand_tracker

# Install steps:
# pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai==0.0.2.1+05055ad0a1038980717ea67505ea7474555d0b0a
# cd ..
# git clone https://github.com/geaxgx/depthai_hand_tracker

# Status: working

pathToProject='../depthai_hand_tracker/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import mediapipe_utils as mpu

useOAKDCam=False
# useOAKDCam=True

def init_model(transform):
    global useOAKDCam
    if transform == 'hands':
        if useOAKDCam:
            ht = HandTracker()
        else:
            ht = HandTracker(input_file='direct')
        ht.init_pipeline()
        return ht, None

def process_image(transform,processing_model,img):
    tracks = []
    try:
        frame = img

        #hand tracking https://github.com/geaxgx/depthai_hand_tracker
        if transform == 'hands':
            tracks, img = processing_model.process_image(img)

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("OAK-D Exception",e)
        pass
                
    return tracks,img

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)

class HandTracker:
    def __init__(self, input_file=None,
                pd_path="models/palm_detection.blob", 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_path="models/hand_landmark.blob",
                lm_score_threshold=0.5):

        self.camera = input_file is None
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.regions = []

        self.seq_num = None
        
        if not self.camera:
            if input_file == "direct":
                self.image_mode = None
            elif input_file.endswith('.jpg') or input_file.endswith('.png') :
                self.image_mode = True
                self.img = cv2.imread(input_file)
                self.video_size = np.min(self.img.shape[:2])
            else:
                self.image_mode = False
                self.cap = cv2.VideoCapture(input_file)
                width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                self.video_size = int(min(width, height))
        
        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Rendering flags
        if self.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = False
            self.show_landmarks = True
            self.show_scores = False
        else:
            self.show_pd_box = True
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_scores = False
        

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        self.pd_input_length = 128

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            # Crop video to square shape (palm detection takes square image as input)
            self.video_size = min(cam.getVideoSize())
            cam.setVideoSize(self.video_size, self.video_size)
            cam.setFps(30)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setVideoSize(self.video_size, self.video_size)
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            # Link video output to host for higher resolution
            cam.video.link(cam_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        p = str(Path(self.pd_path).resolve().absolute())
        print("pd_path",p)
        pd_nn.setBlobPath(p)
        # Increase threads for detection
        # self.pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Palm detection input                 
        if self.camera:
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            cam.preview.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)
        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
       

         # Define hand landmark model
        if self.use_lm:
            print("Creating Hand Landmark Neural Network...")          
            lm_nn = pipeline.createNeuralNetwork()
            p = str(Path(self.lm_path).resolve().absolute())
            print("lm_path",p)
            lm_nn.setBlobPath(p)
            lm_nn.setNumInferenceThreads(1)
            # Hand landmark input
            self.lm_input_length = 224
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)
            # Hand landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        

    
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.regions)
            mpu.rect_transformation(self.regions, self.video_size, self.video_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.video_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.video_size)
                    y = int(kp[1] * self.video_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.video_size+10), int((r.pd_box[1]+r.pd_box[3])*self.video_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = inference.getLayerFp16("Identity_dense/BiasAdd/Add")
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> keep x/w,y/h
            lm.append([lm_raw[3*i]/self.lm_input_length, lm_raw[3*i+1]/self.lm_input_length])
        region.landmarks = lm
    
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
            if self.show_landmarks:
                src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
                dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
                mat = cv2.getAffineTransform(src, dst)
                lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
                lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
                list_connections = [[0, 1, 2, 3, 4], 
                                    [0, 5, 6, 7, 8], 
                                    [5, 9, 10, 11, 12],
                                    [9, 13, 14 , 15, 16],
                                    [13, 17],
                                    [0, 17, 18, 19, 20]]
                lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
                cv2.polylines(frame, lines, False, (255, 0, 0), 2, cv2.LINE_AA)
                for x,y in lm_xy:
                    cv2.circle(frame, (x, y), 6, (0,128,255), -1)
            if self.show_handedness:
                cv2.putText(frame, f"RIGHT {region.handedness:.2f}" if region.handedness > 0.5 else f"LEFT {1-region.handedness:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+20)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if region.handedness > 0.5 else (0,0,255), 2)
            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+90)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
        

    def init_pipeline(self):

        self.device = dai.Device(self.create_pipeline())
        self.device.startPipeline()

        # Define data queues 
        if self.camera:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")
        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")

        self.seq_num = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0

    def process_image(self,img):
        annotated_frame = img
        if self.camera:
            in_video = self.q_video.get()
            # Convert NV12 to BGR
            yuv = in_video.getData().reshape((in_video.getHeight() * 3 // 2, in_video.getWidth()))
            video_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        else:
            if self.image_mode is None:
                vid_frame = img
                height, width, _ = img.shape
                self.video_size = int(min(width, height))
            elif self.image_mode:
                vid_frame = self.img
            else:
                ok, vid_frame = self.cap.read()
                if not ok:
                    # print("not OK video frame")
                    return [], img #break

            h, w = vid_frame.shape[:2]
            dx = (w - self.video_size) // 2
            dy = (h - self.video_size) // 2
            video_frame = vid_frame[dy:dy+self.video_size, dx:dx+self.video_size]
            frame_nn = dai.ImgFrame()
            frame_nn.setSequenceNum(self.seq_num)
            frame_nn.setWidth(self.pd_input_length)
            frame_nn.setHeight(self.pd_input_length)
            frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))

            self.q_pd_in.send(frame_nn)

            self.seq_num += 1

        annotated_frame = video_frame.copy()

        inference = self.q_pd_out.get()
        self.pd_postprocess(inference)
        self.pd_render(annotated_frame)

        # Hand landmarks
        if self.use_lm:
            for i,r in enumerate(self.regions):
                img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                self.q_lm_in.send(nn_data)
            
            # Retrieve hand landmarks
            for i,r in enumerate(self.regions):
                inference = self.q_lm_out.get()
                self.lm_postprocess(r, inference)
                self.lm_render(annotated_frame, r)
        
        return self.regions,annotated_frame
