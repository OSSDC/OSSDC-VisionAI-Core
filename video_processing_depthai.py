
import traceback
import queue
from pathlib import Path
import cv2

import sys
import argparse
import os

import numpy as np
import depthai as dai
import time
from threading import Thread


depthAiImg = None

debug=True

useOAKDCam=False
threadStarted = False
t = None

unconsumedFrameCount = 0

from pynput.keyboard import Key, Listener
  
diff = 0

def on_press(k):
    global diff,t
    if str(k)[1] == '=':
         if diff <= -10:
              diff=diff+10
    elif str(k)[1] == '-':
         diff=diff-10
    elif str(k)[1] == 'q':
         t.stop()
    print('key', k, 'diff', diff)
    
  
listener = Listener(
    on_press=on_press)
listener.start()

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    # xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    # cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    # xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb')

    # cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb'] #['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_mono_cam_pipeline():
    print("Creating pipeline: MONO CAMS -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam_left   = pipeline.createMonoCamera()
    cam_right  = pipeline.createMonoCamera()
    xout_left  = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()

    cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    for cam in [cam_left, cam_right]: # Common config
        cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        #cam.setFps(20.0)

    xout_left .setStreamName('left')
    xout_right.setStreamName('right')

    cam_left .out.link(xout_left.input)
    cam_right.out.link(xout_right.input)

    streams = ['left', 'right']

    return pipeline, streams


def test_pipeline(a,b,c):
    global depthAiImg, diff, unconsumedFrameCount
    # global clientsocket
    # print("a,b,transform",a,b,transform)
    transform=a+b+c
    if transform == 'rgb':
        pipeline, streams = create_rgb_cam_pipeline()
    else:
        pipeline, streams = create_mono_cam_pipeline()
    # pipeline, streams = create_stereo_depth_pipeline(source_camera)

    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()

        in_streams = []
        # if not source_camera:
        #     # Reversed order trick:
        #     # The sync stage on device side has a timeout between receiving left
        #     # and right frames. In case a delay would occur on host between sending
        #     # left and right, the timeout will get triggered.
        #     # We make sure to send first the right frame, then left.
        #     in_streams.extend(['in_right', 'in_left'])
        in_q_list = []
        inStreamsCameraID = []
        for s in in_streams:
            q = device.getInputQueue(s)
            in_q_list.append(q)
            inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]

        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            print("stream found",s)
            #if s in ['disparity']:
                #print("stream added",s)
            q = device.getOutputQueue(s, 8, blocking=True)
            q_list.append(q)

        # Need to set a timestamp for input frames, for the sync stage in Stereo node
        timestamp_ms = 0
        index = 0
        
        img = None
        frameCount=0
        prevFrameCount=0
        prevTime = time.time()

        imgs = {}
        imgs['rgb'] = None
        imgs['left'] = None
        imgs['right'] = None
        imgs['disparity'] = None

        black = np.zeros((960,2560), dtype = "uint8")
        h,w = black.shape
        #diff = 0
        while True:
            if unconsumedFrameCount>120: #stop if no client consuming
                break
            for q in q_list:
                name  = q.getName()
                if name in ['rgb']:
                    image = q.get()
                    frame = image.getCvFrame()
                    depthAiImg = frame
                    # depthAiImg = cv2.resize(depthAiImg,(1440,))
                    # depthAiImg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

                elif name in ['left', 'right']:
                    image = q.get()
                    frame = image.getCvFrame()
                    imgs[name] = frame
                    if imgs['left'] is not None and imgs['right'] is not None:
                        # temp = np.concatenate((imgs['left'], imgs['right']), axis = 1)
                        # black[119:(h-120-1), 0:w] = temp
                        # depthAiImg = black
                        # depthAiImg = cv2.cvtColor(depthAiImg,cv2.COLOR_GRAY2RGB)

                        green_channel = imgs['left']
                        red_channel = imgs['right']
                        #red_channel = imgs['left']
                        #green_channel = imgs['right']

                        ih,iw = red_channel.shape
                        # print(h,w, ih,iw)
                        # temp = np.concatenate((green_channel,red_channel), axis = 1)
                        # h,w = black.shape

                        #diff = -40 #change this to fit your Phone and VR headset size

                        black[h//2-ih//2 : h//2 + ih//2, w//2 - iw - diff : w//2 - diff] = green_channel #red_channel #green_channel
                        black[h//2-ih//2 : h//2 + ih//2, w//2 + diff : w//2 + iw + diff] = red_channel
                        # black[119:(h-121), 119:w-121] = temp
                        depthAiImg = black
                        # depthAiImg = cv2.resize(depthAiImg,(iw//2,ih//2))
                        depthAiImg = cv2.cvtColor(depthAiImg,cv2.COLOR_GRAY2RGB) 

                        imgs['left'] = None
                        imgs['right'] = None                
            
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            frameCount+=1
            unconsumedFrameCount=unconsumedFrameCount+1
            delta = time.time() - prevTime
            if delta > 1:
                realFPS = (frameCount-prevFrameCount)/delta
                print("realFps",realFPS,'diff',diff)
                if depthAiImg is not None:
                    # cv2.imshow('depthAiImg', depthAiImg)
                    print("depthAiImg.shape",depthAiImg.shape)
                prevFrameCount = frameCount
                prevTime = time.time()      


def init_model(transform):
    global newTransform
    # print('init_model',transform)
    if transform in ['rgb', 'sbs']:
        return None, None
    return None, None

def process_image(transform,processing_model,img):
    global useOAKDCam, depthAiImg, threadStarted, t, unconsumedFrameCount
    # print('process_image',transform)
    tracks = []
    try:
        frame = img
        if depthAiImg is not None and transform in ['rgb', 'sbs']:
            unconsumedFrameCount=unconsumedFrameCount-1
            img = depthAiImg
        if not threadStarted:
            threadStarted = True
            try:
                t = Thread(target=test_pipeline, args=(str(transform)))
                t.start()
            except Exception as e:
                print(e)

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("DepthAI Exception",e)
        pass
                
    return tracks,img
