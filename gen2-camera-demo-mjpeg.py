#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
import time

'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''

import numpy as np
import open3d as o3d

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = None

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.isstarted = False

    def rgbd_to_projection(self, depth_map, rgb, is_rgb):
        self.depth_map = depth_map
        self.rgb = rgb
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)
        # TODO: query frame shape to get this, and remove the param 'is_rgb'
        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        if self.pcl is None:
            self.pcl = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
            self.pcl.points = pcd.points
            self.pcl.colors = pcd.colors
        return self.pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self.vis.add_geometry(self.pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.vis.add_geometry(origin)
            self.isstarted = True
        else:
            self.vis.update_geometry(self.pcl)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()


parser = argparse.ArgumentParser()
parser.add_argument("-pcl", "--pointcloud", help="enables point cloud convertion and visualization", default=False, action="store_true")
parser.add_argument("-static", "--static_frames", default=False, action="store_true",
                    help="Run stereo on static frames passed from host 'dataset' folder")
args = parser.parse_args()

point_cloud    = args.pointcloud   # Create point cloud visualizer. Depends on 'out_rectified'

# StereoDepth config options. TODO move to command line options
source_camera  = not args.static_frames
out_depth      = point_cloud  # Disparity by default
out_rectified  = False #not point_cloud   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = True  # Closer-in minimum depth, disparity range is doubled
subpixel = not extended   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
# median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
confidenceThreshold = 200 #default 200

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

# TODO add API to read this from device / calib data
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]

pcl_converter = None
if point_cloud:
    if out_rectified:
        # try:
        #     from projector_3d import PointCloudVisualizer
        # except ImportError as e:
        #     raise ImportError(f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")
        pcl_converter = PointCloudVisualizer(right_intrinsic, 1280, 720)
    else:
        print("Disabling point-cloud visualizer, as out_rectified is not set")

import socket

# To monitor the output from video processing run on your PC this command:
# ffplay -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
ip = 'localhost' #replace with your PC IP where ffplay runs
ip = None #comment to activate above IP
# ip = '192.168.1.117'

clientsocket = None
if ip is not None:
  try:
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.settimeout(5)
    clientsocket.connect((ip,45654)) #the target ip is where the ffplay is listening
  except Exception as e:
    print(e)
    clientsocket = None

last_rectif_right = None
last_rectif_left = None


def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()

    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    cam.preview.link(xout_preview.input)
    cam.video  .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

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
    stereo.setConfidenceThreshold(confidenceThreshold)
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

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image):
    global last_rectif_right, last_rectif_left
    baseline = 75 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32;
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video': # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        # disp = wls(last_rectif_left,disp, disp)
        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            # frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        if pcl_converter is not None:
            if 0: # Option 1: project colorized disparity
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pcl_converter.rgbd_to_projection(depth, frame_rgb, True)
            else: # Option 2: project rectified right
                pcl_converter.rgbd_to_projection(depth, last_rectif_right, False)
            pcl_converter.visualize_pcd()

        
    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
            if name == 'rectified_right':
                last_rectif_right = frame
            else:
                last_rectif_left = frame
    return frame


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

def wls(imgL, displ, dispr):
    
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    return filteredImg

def depth_map(imgL, imgR):

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    return wls(imgL, displ, dispr)


def test_pipeline():
    global clientsocket, last_rectif_right
   #pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    pipeline, streams = create_stereo_depth_pipeline(source_camera)

    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()

        in_streams = []
        if not source_camera:
            # Reversed order trick:
            # The sync stage on device side has a timeout between receiving left
            # and right frames. In case a delay would occur on host between sending
            # left and right, the timeout will get triggered.
            # We make sure to send first the right frame, then left.
            in_streams.extend(['in_right', 'in_left'])
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
        imgs['rectified_left'] = None
        imgs['rectified_right'] = None
        depthAiImg = None

        skipPairs = 0

        while True:
            # Handle input streams, if any
            # if in_q_list:
            #     dataset_size = 2  # Number of image pairs
            #     frame_interval_ms = 33
            #     for i, q in enumerate(in_q_list):
            #         name = q.getName()
            #         path = 'dataset/' + str(index) + '/' + name + '.png'
            #         data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
            #         tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
            #                                     milliseconds = timestamp_ms % 1000)
            #         img = dai.ImgFrame()
            #         img.setData(data)
            #         img.setTimestamp(tstamp)
            #         img.setInstanceNum(inStreamsCameraID[i])
            #         img.setType(dai.ImgFrame.Type.RAW8)
            #         img.setWidth(1280)
            #         img.setHeight(720)
            #         q.send(img)
            #         if timestamp_ms == 0:  # Send twice for first iteration
            #             q.send(img)
            #         print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
            #     timestamp_ms += frame_interval_ms
            #     index = (index + 1) % dataset_size
            #     if 1: # Optional delay between iterations, host driven pipeline
            #         sleep(frame_interval_ms / 1000)
            # Handle output streams
            for q in q_list:
                name  = q.getName()
                image = q.get()
                #print("Received frame:", name)
                # Skip some streams for now, to reduce CPU load
                # if name in ['left', 'right', 'depth']: continuelast_rectif_right
                if point_cloud:
                    frame = convert_to_cv2_frame(name, image)
                if not point_cloud and name in ['rectified_right']:# , 'rectified_right']:
                    last_rectif_right = image.getCvFrame()
                elif not point_cloud and name in ['rectified_left']:# , 'rectified_right']:
                    last_rectif_left = image.getCvFrame()
                    # imgs[name] = frame  
                elif point_cloud and name in ['depth']:
                    # frame = convert_to_cv2_frame(name, image)
                    img = frame
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    cv2.imshow(name, img)

                elif not point_cloud and name in ['disparity']:
                    # frame = image.getCvFrame()
                    frame = convert_to_cv2_frame(name, image)
                    img = frame
                    # print("img.shape",img.shape)
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                    # kernel = np.ones((3,3), np.uint8)       # set kernel as 3x3 matrix from numpy
                    # #Create erosion and dilation image from the original image
                    # erosion_image = cv2.erode(img, kernel, iterations=1)
                    # img = erosion_image
                    # dilation_image = cv2.dilate(img, kernel, iterations=1)
                    # img = dilation_image
                    cv2.imshow(name, img)
                    try:
                        if clientsocket is not None:
                            img = frame #img.to_ndarray(format="bgr24")
                            data = cv2.imencode('.jpg', img)[1].tobytes()
                            clientsocket.send(data)          
                            # cv2.imshow(self.transformLabel, img)
                            # k = cv2.waitKey(1) & 0xff
                            # if k == 27 : 
                            #     break

                    except Exception as e:
                        print(e)
                        clientsocket = None
                        pass   
                
            if not point_cloud and last_rectif_right is not None and last_rectif_right is not None:
                if skipPairs==0:
                    depthAiImg = depth_map(last_rectif_left,last_rectif_right)
                    # depthAiImg = cv2.cvtColor(depthAiImg, cv2.COLOR_GRAY2BGR)
                    # depthAiImg = cv2.applyColorMap(depthAiImg, cv2.COLORMAP_JET)
                    # cv2.imshow('rectified_left', last_rectif_left)
                    # cv2.imshow('rectified_right', last_rectif_right)
                    img = depthAiImg
                    last_rectif_left = None
                    last_rectif_right = None
                    skipPairs = 4 #to increase responsevness
                else:
                    skipPairs = skipPairs - 1

            if depthAiImg is not None:
                cv2.imshow('depth_host', depthAiImg)

            frameCount+=1
            delta = time.time() - prevTime
            if delta > 1:
                realFPS = (frameCount-prevFrameCount)/delta
                print("realFps",realFPS)
                if img is not None:
                    print("img.shape",img.shape)
                prevFrameCount = frameCount
                prevTime = time.time()                  

            if cv2.waitKey(1) == ord('q'):
                break


test_pipeline()
