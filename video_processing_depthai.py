import traceback
# import queue
from pathlib import Path
import cv2

# import sys
# import argparse
# import os

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

out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO


try:
    from pynput.keyboard import Key, Listener
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
except:
    pass

diff = 0



""" Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
# SGBM Parameters -----------------
window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

left_matcher = cv2.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=window_size,
    P1=8 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.3
visual_multiplier = 6

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)

wls_filter.setSigmaColor(sigma)
    
def depth_map(imgL, imgR):

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

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


def create_stereo_depth_pipeline(from_camera=True):
    print("Creating Stereo Depth pipeline: ", end='')
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()

    if from_camera:
        cam_left      = pipeline.createMonoCamera()
        cam_right     = pipeline.createMonoCamera()
    else:
        cam_left      = pipeline.createXLinkIn()
        cam_right     = pipeline.createXLinkIn()
    stereo            = pipeline.createStereoDepth()
    xout_left         = pipeline.createXLinkOut()
    xout_right        = pipeline.createXLinkOut()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rectif_left  = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()

    if from_camera:
        cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]: # Common config
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            #cam.setFps(20.0)
    else:
        cam_left .setStreamName('in_left')
        cam_right.setStreamName('in_right')

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
        #stereo.loadCalibrationFile(path)
        pass
    else:
        stereo.setEmptyCalibration() # Set if the input frames are already rectified
        stereo.setInputResolution(1280, 720)

    xout_left        .setStreamName('left')
    xout_right       .setStreamName('right')
    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rectif_left .setStreamName('rectified_left')
    xout_rectif_right.setStreamName('rectified_right')

    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    stereo.syncedLeft    .link(xout_left.input)
    stereo.syncedRight   .link(xout_right.input)
    stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    if out_rectified:
        stereo.rectifiedLeft .link(xout_rectif_left.input)
        stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ['left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

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
    elif transform == 'map':
        pipeline, streams = create_stereo_depth_pipeline()
    else:
        pipeline, streams = create_mono_cam_pipeline()
    # pipeline, streams = create_stereo_depth_pipeline(source_camera)

    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()

        # in_streams = []
        # # if not source_camera:
        # #     # Reversed order trick:
        # #     # The sync stage on device side has a timeout between receiving left
        # #     # and right frames. In case a delay would occur on host between sending
        # #     # left and right, the timeout will get triggered.
        # #     # We make sure to send first the right frame, then left.
        # #     in_streams.extend(['in_right', 'in_left'])
        # in_q_list = []
        # inStreamsCameraID = []
        # for s in in_streams:
        #     q = device.getInputQueue(s)
        #     in_q_list.append(q)
        #     inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]

        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            print("stream found",s)
            #if s in ['disparity']:
                #print("stream added",s)
            q = device.getOutputQueue(s, 8, blocking=True)
            q_list.append(q)

        # # Need to set a timestamp for input frames, for the sync stage in Stereo node
        # timestamp_ms = 0
        # index = 0
        
        img = None
        frameCount=0
        prevFrameCount=0
        prevTime = time.time()

        imgs = {}
        imgs['rgb'] = None
        imgs['left'] = None
        imgs['right'] = None
        imgs['disparity'] = None
        imgs['rectified_left'] = None
        imgs['rectified_right'] = None

        black = np.zeros((960,2560), dtype = "uint8")
        h,w = black.shape
        #diff = 0
        while True:
            # if unconsumedFrameCount>120: #stop if no client consuming
            #     break
            for q in q_list:
                name  = q.getName()
                image = q.get()
                frame = image.getCvFrame()
                if name in ['rgb']:
                    depthAiImg = frame
                    # depthAiImg = cv2.resize(depthAiImg,(1440,))
                    # depthAiImg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
                elif name in ['left', 'right', 'rectified_left', 'rectified_right']:
                    imgs[name] = frame
                    
                    if transform == 'map':
                        imgLeft = imgs['rectified_left']
                        imgRight = imgs['rectified_right']
                    else:
                        imgLeft = imgs['left']
                        imgRight = imgs['right']

                    if imgLeft is not None and imgRight is not None:
                        if transform == 'map':
                            depthAiImg = depth_map(imgLeft,imgRight)
                            depthAiImg = cv2.cvtColor(depthAiImg, cv2.COLOR_GRAY2BGR)
                            depthAiImg = cv2.applyColorMap(depthAiImg, cv2.COLORMAP_JET)
                            imgs['rectified_left'] = None
                            imgs['rectified_right'] = None                
                        else:
                            # temp = np.concatenate((imgs['left'], imgs['right']), axis = 1)
                            # black[119:(h-120-1), 0:w] = temp
                            # depthAiImg = black
                            # depthAiImg = cv2.cvtColor(depthAiImg,cv2.COLOR_GRAY2RGB)

                            green_channel = imgLeft
                            red_channel = imgRight
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
    if transform in ['rgb', 'sbs', 'map']:
        return None, None
    return None, None

def process_image(transform,processing_model,img):
    global useOAKDCam, depthAiImg, threadStarted, t, unconsumedFrameCount
    # print('process_image',transform)
    tracks = []
    try:
        frame = img
        if depthAiImg is not None and transform in ['rgb', 'sbs', 'map']:
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
