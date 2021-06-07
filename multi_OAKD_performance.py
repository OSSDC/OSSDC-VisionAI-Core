import cv2
import numpy as np
import depthai as dai
import time
from threading import Thread

import socket

# Start defining a pipeline
pipeline = dai.Pipeline()


# Define a source - color camera
camRgb = pipeline.createColorCamera()
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
# camRgb.setInterleaved(True)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

camRgb.setVideoSize(1280,720)
#camRgb.setVideoSize(1920,1080)
#camRgb.setVideoSize(3840,2160)

# Create output
xoutVideo = pipeline.createXLinkOut()
xoutVideo.setStreamName("video")
xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

camRgb.video.link(xoutVideo.input)


# To monitor the output from video processing run on your PC this command:
# ffplay -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
ip = 'localhost' #replace with your PC IP where ffplay runs
ip = None #comment to activate above IP
ip ='192.168.1.6'

clientsocket = None
if ip is not None:
  try:
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.settimeout(5)
    clientsocket.connect((ip,45654)) #the target ip is where the ffplay is listening
  except Exception as e:
    print(e)
    clientsocket = None


def run_on_device(pipeline, device_info, deviceId):
    global clientsocket
    with dai.Device(pipeline, device_info) as device:
        print("Starting pipeline on",deviceId)
        # Start pipeline
        device.startPipeline()

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="video", maxSize=1, blocking=False)
        global img
        img = None
        frameCount=0
        prevFrameCount=0
        prevTime = time.time()
        while True:
            inRgb = qRgb.tryGet() 
            if inRgb is None:
                continue

            img = inRgb.getCvFrame()

            #if img is not None:
                # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                # gray = np.float32(gray)
                # dst = cv2.cornerHarris(gray,2,3,0.04)

                # #result is dilated for marking the corners, not important
                # dst = cv2.dilate(dst,None)

                # # Threshold for an optimal value, it may vary depending on the image.
                # img[dst>0.01*dst.max()]=[0,0,255]

                #cv2.imshow('dst',img)

            try:
                if clientsocket is not None and img is not None:
                    # img = img.to_ndarray(format="bgr24")
                    rows, cols, _ = img.shape
                    #scale = 4
                    #img = cv2.resize(img,(cols//scale, rows//scale))
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

                

            frameCount+=1
            delta = time.time() - prevTime
            if delta > 1:
                realFPS = (frameCount-prevFrameCount)/delta
                print(deviceId,"realFps",realFPS)
                if img is not None:
                    print("img.shape",img.shape)
                prevFrameCount = frameCount
                prevTime = time.time()
                            
           # if cv2.waitKey(1) == ord('q'):
           #     break



for device in dai.Device.getAllAvailableDevices():
    print(f"{device.getMxId()} {device.state}")
    found, device_info = dai.Device.getDeviceByMxId(device.getMxId())

    if not found:
        continue

    try:
        t = Thread(target = run_on_device, args =(pipeline, device_info, device.getMxId()))
        t.start()

    except Exception as e:
        print(e)
