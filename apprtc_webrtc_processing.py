import argparse
import asyncio
import logging
import os
import random
import numpy as np
import cv2
from av import VideoFrame
import traceback 
import subprocess as sp

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except:
    pass

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceCandidate,
    RTCIceServer,
    VideoStreamTrack,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
# from signaling_race import BYE, RaceOssdcSignaling, object_from_string, object_to_string, sendSubscribeMessage,sendUnSubscribeMessage, roomName, sendMessage,droneRoomName,sio
from aiortc.contrib.signaling import BYE, ApprtcSignaling

import sys
import argparse

import time
import json

import socket

import youtube_dl

import subprocess

# from twitchstream.outputvideo import TwitchBufferedOutputStream

video_processing_module = []

debug=True

def debug_print(*argv):
    if(debug):
        print(*argv)

# To monitor the output from video processing run on your PC this command:
# ffplay -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
ip = 'localhost' #replace with your PC IP where ffplay runs
# ip = '192.168.1.6'
# ip = None #comment to activate above IP

clientsocket = None
if ip is not None:
  try:
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.settimeout(5)
    clientsocket.connect((ip,45654)) #the target ip is where the ffplay is listening
  except Exception as e:
    debug_print(e)
    clientsocket = None


twitchStream = None

def getTwitchStream(streamKey, width, height):
    twitchStream = TwitchBufferedOutputStream(
            twitch_stream_key=streamKey,
            width=width,
            height=height,
            fps=30.,
            verbose=True,
            enable_audio=False)
    return twitchStream

# infoColor = (255,255,255)
infoColor1 = (0,255,0)
infoColor2 = (0,255,255)
infoColor = infoColor1

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, signaling, model, skipFrames=False, skipFramesCnt=0):
        super().__init__()  # don't forget this!
        # self.transform = transform
        self.transformLabel = None
        self.transform = []
        self.processing_model = []
        for i in range(len(transform)):
            (alg,subAlg) = transform[i]
            self.processing_model.append(model[i])
            self.transform.append(subAlg)        
            if self.transformLabel is None: 
                self.transformLabel = alg 
            else:
                self.transformLabel = self.transformLabel + "+" + alg 
            # print(video_processing_module[i],self.transform[i],self.processing_model[i])        
        # print(video_processing_module[0],transform[0])        
        # video_processing_module[str(0)].init_model(transform[0])        

        self.track = track
        self.scale = 1
        self.skipFrames = skipFrames #False #use it for Youtube streaming
        self.skipFrames = False
        self.prevTime = time.time()
        self.starttime1 = time.time()
        self.colors = "Not computed"
        self.signaling = signaling
        self.frameProcessedCount = 0
        self.frameCount = 0
        self.prevFrameCount = 0
        self.realFPS = 0
        self.skipFramesCnt = skipFramesCnt #to skip frames at begining of video
        self.imgM = {}
        self.trackPts = {}
        self.prevGoodFrame = None     
        self.saveOrigAndProc = False

    async def recv(self):
        global clientsocket, twitchStream, infoColor, video_processing_module
        frame = await self.track.recv()
        self.frameCount=self.frameCount+1

        if self.skipFramesCnt>0:
          while not self.track._queue.empty():
              frame = await self.track.recv()
              self.frameCount=self.frameCount+1
              if self.skipFramesCnt==0:
                break
              self.skipFramesCnt=self.skipFramesCnt-1
          return frame
        
        new_frame = frame

        # Consume all available frames.
        # If we don't do this, we'll bloat indefinitely.
        if self.skipFrames:
          while not self.track._queue.empty():
              frame = await self.track.recv() 
              self.frameCount=self.frameCount+1

        self.frameProcessedCount=self.frameProcessedCount+1

        timer = cv2.getTickCount()

        try:
        
            img = frame.to_ndarray(format="bgr24")

            rows, cols, _ = img.shape
            #debug_print('before',img.shape)

            if self.scale!=1:
                img = cv2.resize(img,(cols//self.scale, rows//self.scale))
                rows, cols, _ = img.shape

            h, w, _ = img.shape
            rows, cols, _ = img.shape

            y = h//3
            x = w//3

    #         img = img[y+200:y+200+y, x:x+x]
    #         rows, cols, _ = img.shape        
    #         h, w, _ = img.shape

    #         y = h//3
    #         x = w//3


            #img = cv2.pyrDown(img)
            imgOut = None
            # i = 0
            for i in range(len(self.transform)):
                imgIn = img.copy()
                self.trackPts[i], self.imgM[i] = video_processing_module[i].process_image(self.transform[i],self.processing_model[i],imgIn)
                if i>0:
                    imgOut = cv2.addWeighted(imgOut, 1/len(self.transform), self.imgM[i], 1/len(self.transform), 0.0)
                else:
                    imgOut = self.imgM[i]
                # i=i+1

            timeVal = str(timer)
            transformAndTime = self.transformLabel+'-'+timeVal
            if self.saveOrigAndProc:
                # timeVal = str(frame.time_base)
                # print(transformAndTime)
                cv2.imwrite('./saved/orig/img-'+transformAndTime+'.png',img)
                cv2.imwrite('./saved/proc/img-'+transformAndTime+'.png',imgOut)

            # print(imgOut.shape,len(self.transform))
            trackingPoints = self.trackPts[len(self.transform)-1]
            img = imgOut #self.imgM[len(self.transform.keys())-1]

            y1 = y

            if 1==2: # for robot control - works with MiDaS for now
                crop_img = img[y:y+y, x:x+x]

                r, c, _ = crop_img.shape


                if 1==1: #robot control
                    avg_color_per_row = np.average(crop_img, axis=0)
                    avg_color = np.average(avg_color_per_row, axis=0)
                    #debug_print(avg_color)

                    pix_total = 1
                    color_B = avg_color[0]
                    color_G = avg_color[1]
                    color_R = avg_color[2]
                    color_N = 1
                    self.colors = ['Blue: {:.2f}'.format(color_B/pix_total), 'Green: {:.2f}'.format(color_G/pix_total), 'Red: {:.2f}'.format(color_R/pix_total)] # + ', Gray: ' + str(color_N/pix_total)

                    debug_print(self.colors)
                    if (time.time() - self.starttime1) > 0.1:
                        self.starttime1 = time.time()
                        if (color_B/pix_total)>200:
                            msg = '{"setmotor":[30,30,100,'+ '1605574844705' + ']}'
                            jsonmsg = json.loads(msg)
                            await self.signaling.send(jsonmsg);
                        else:
                            msg = '{"setmotor":[-30,30,50,'+ '1605574844705' + ']}'
                            jsonmsg = json.loads(msg)
                            await self.signaling.send(jsonmsg);
    #                     debug_print('sent message: ',jsonmsg)

            #         cv2.rectangle(img, (x,y), (x+x,y+y), (50,170,50), 2)
                    cv2.rectangle(img, (x,y), (x+x,y+y), (0,0,0), 2)

                    cv2.putText(img, self.colors[0], (10,y1+125+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                    cv2.putText(img, self.colors[1], (10,y1+150+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                    cv2.putText(img, self.colors[2], (10,y1+175+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)

    #         img = cv2.resize(img,(cols//4, rows//4))
    #         y = h//3
    #         x = w//3
    #         y1 = y+25                

            if True:
                cv2.putText(img, "Alg: "+self.transformLabel, (10,y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "ImagSize: "+str(w)+"x"+str(h), (10,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "FramCnt: "+str(self.frameCount), (10,y1+75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "FramProcCnt: "+str(self.frameProcessedCount), (10,y1+100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "TrkPt: "+str(len(trackingPoints)), (10,y1+125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)

                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

                cv2.putText(img, "ProcFPS : " + str(int(fps)), (10,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "RealFPS : " + str(int(self.realFPS)), (10,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)
                cv2.putText(img, "Race AI with us at OSSDC.org - Open Source Self Driving Initiative ", (10,h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, infoColor, 2)

    #         img = cv2.resize(img,(cols//3, rows//3))

            delta = time.time() - self.prevTime
            if delta > 1:
                self.realFPS = (self.frameCount-self.prevFrameCount)/delta
                print(fps,self.realFPS, img.shape, transformAndTime)
                self.prevFrameCount = self.frameCount
                self.prevTime = time.time()
                if infoColor == infoColor1:
                    infoColor = infoColor2
                else:
                    infoColor = infoColor1

            try:
                if twitchStream is not None:
                    ret2, frame2 = cv2.imencode('.png', img)
                    twitchStream.stdin.write(frame2.tobytes())
                #     if twitchStream.get_video_frame_buffer_state() < 30:
                #         # frame = np.random.rand(480, 640, 3)
                #         imgROI = img[0:359, 0:639]
                #         twitchStream.send_video_frame(imgROI)
            except Exception as e:
                twitchStream = None
                track = traceback.format_exc()
                print("Twitch exception",e)
                pass    

            try:
                if clientsocket is not None:
                    # img = img.to_ndarray(format="bgr24")
                    data = cv2.imencode('.jpg', img)[1].tobytes()
                    clientsocket.send(data)          
                    # cv2.imshow(self.transformLabel, img)
                    # k = cv2.waitKey(1) & 0xff
                    # if k == 27 : 
                    #     break

            except Exception as e:
                debug_print(e)
                clientsocket = None
                pass     

            # new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            try:
                new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                self.prevGoodFrame = new_frame
            except Exception as e:
                if self.prevGoodFrame is not None:
                    new_frame = self.prevGoodFrame
                pass   
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base 
        except Exception as e1:
            track = traceback.format_exc()
            print(track)
            print("RaceOSSDC",e1)
            pass

        return new_frame

async def run(pc, player, recorder, signaling, transform, model, skipFrames, skipFramesCnt):
    def add_tracks():
        if player and player.audio:
            pc.addTrack(player.audio)

        if player and player.video:
            # pc.addTrack(player.video)
            local_video = VideoTransformTrack(player.video, transform=transform, signaling=signaling, model=model, skipFrames=skipFrames, skipFramesCnt=skipFramesCnt)
            pc.addTrack(local_video)
        else:
            # localTrack =  VideoTransformTrack(None)
            # pc.addTrack(localTrack)
            pc.addTransceiver('video','sendrecv') #this is the trick to echo webcam back

            # pc.addTransceiver('video','sendrecv') #this is the trick to echo webcam back

    @pc.on("track")
    def on_track(track):
        print("Track %s received" % track.kind)
        if track.kind == "video":
            if not(player and player.video):
                local_video = VideoTransformTrack(
                    track, transform=transform, signaling=signaling, model=model
                )
                pc.addTrack(local_video)

            # @track.on("ended")
            # async def on_ended():
            # debug_print("track ended")
            # signaling.trackEnded=True
        recorder.addTrack(track)

    # connect to websocket and join
    params = await signaling.connect()

    if params["is_initiator"] == "true":
        # send offer
        add_tracks()
        await pc.setLocalDescription(await pc.createOffer())
        await signaling.send(pc.localDescription)

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)
            await recorder.start()

            if obj.type == "offer":
                # send answer
                add_tracks()
                await pc.setLocalDescription(await pc.createAnswer())
                await signaling.send(pc.localDescription)
        elif isinstance(obj, RTCIceCandidate):
            await pc.addIceCandidate(obj)
        elif obj is BYE:
            print("Exiting")
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RaceOSSDC")
    
    parser.add_argument("--play-from", help="Read the media from a file and sent it."),
    parser.add_argument("-rec", help="Write received media to a file."),
    parser.add_argument('-t','--transform', type=str)
    parser.add_argument('--skipFrames', type=str, default=None)
    parser.add_argument('--videoUrl', type=str, default=None,  nargs='?', const=None)
    parser.add_argument('--skipFramesCnt', type=str, default=0, nargs='?', const=0)
    parser.add_argument('--twitchStreamKey', type=str, default=None,  nargs='?', const=None )
    parser.add_argument('-r','--room', type=str)

       
    args, unknown = parser.parse_known_args()

    if args.room:
        roomName = args.room

    transform = None
    if args.transform:
        transform = args.transform

    print("Room name:",roomName)
    import importlib

            
    algos = transform.split("+") #alows blending algos
    transform = []
    model = []
    modules = []
    i = 0
    for alg in algos:
        module_name = alg.split(".")
        if len(module_name) == 2:
            subAlg = module_name[1]
            module_name = module_name[0]
        else:
            module_name = module_name[0]
            subAlg = module_name
        debug_print("We will apply this transform:",subAlg, "from module:",module_name)
        moduleIndex = -1
        if module_name in modules:
            moduleIndex = modules.index(module_name)
        if moduleIndex == -1:
            video_processing_module.append(importlib.import_module("video_processing_"+module_name))
        else:
            video_processing_module.append(video_processing_module[moduleIndex])

        modules.append(module_name)
        # importlib.exec_module(video_processing_module[i])

        # video_processing_module.append(vpm)
        print('video_processing_module',video_processing_module[i], subAlg)
        m, args1 = video_processing_module[i].init_model(subAlg)
        transform.append((alg, subAlg))
        model.append(m)
        i=i+1

    # create signaling and peer connection
    args.room = roomName

    videoUrl = None 

    skipFrames = False
    skipFramesCnt = 0

    if args.skipFramesCnt:
        skipFramesCnt = int(args.skipFramesCnt)

    if args.videoUrl:
        videoUrl = args.videoUrl

    if videoUrl is not None:
        if "youtube.com" in videoUrl:
            #install youtube-dl for this to work: pip install youtube-dl
            command = "youtube-dl -f 'bestvideo[height<1000]' -g '"+videoUrl+"'" 
            # command = "youtube-dl -f 'bestvideo' -g '"+videoUrl+"'" 
            videoUrl = subprocess.check_output(command, shell = True).decode("utf-8").strip()
            args.play_from = videoUrl
        elif "direct" == videoUrl:
            skipFrames = True
        else:
            args.play_from = videoUrl
    else:
        skipFrames = True

    print('videoUrl=',videoUrl)

    # create signaling and peer connection
    # signaling = RaceOssdcSignaling(args.room)
    signaling = ApprtcSignaling(args.room)

    
    configuration = RTCConfiguration()
    
    stunServer = RTCIceServer("stun:race.ossdc.org:5349")

    turnServer = RTCIceServer("turn:race.ossdc.org:5349")
    # url: 'turn:race.ossdc.org:5349',
    turnServer.username = "testturn"
    turnServer.credential = roomName

    configuration.iceServers = []
    configuration.iceServers.append(stunServer)
    #configuration.iceServers.append(turnServer)

    pc = RTCPeerConnection(configuration=configuration)

    # create media source
    if args.play_from:
        if "/dev/" in args.play_from:
            # options = {"framerate": "30", "video_size": "640x480"}
            player = MediaPlayer(args.play_from, format="v4l2")#,options=options)
            skipFrames = True
        else:
            player = MediaPlayer(args.play_from)
    else:
        player = None

    # create media sink
    if args.rec:
        print('record to:',args.rec)
        recorder = MediaRecorder(args.rec)
    else:
        recorder = MediaBlackhole()

    if args.skipFrames:
        if args.skipFrames == 'true':
            skipFrames = True
        else:
            skipFrames = False

    loop = asyncio.get_event_loop()

    if args.twitchStreamKey:
        twitchStreamKey = args.twitchStreamKey
        # twitchStream = getTwitchStream(twitchStreamKey,640,360)
        sizeStr = "486x1062"
        fps = 30
        rtmp_server = "rtmp://yto.contribute.live-video.net/app/"+twitchStreamKey
        
        command = ['ffmpeg',
                '-re',
                '-s', sizeStr,
                '-r', str(fps),  # rtsp fps (from input server)
                '-i', '-',
                
                # You can change ffmpeg parameter after this item.
                '-pix_fmt', 'yuv420p',
                '-r', '30',  # output fps
                '-g', '50',
                '-c:v', 'libx264',
                '-b:v', '2M',
                '-bufsize', '64M',
                '-maxrate', "4M",
                '-preset', 'veryfast',
                '-rtsp_transport', 'tcp',
                '-segment_times', '5',
                #    '-f', 'rtsp',
                #    rtsp_server]
                '-f', 'flv',
                rtmp_server]

        twitchStream = sp.Popen(command, stdin=sp.PIPE)

    try:
        loop.run_until_complete(
            run(pc=pc, player=player, recorder=recorder, signaling=signaling,transform=transform, model=model, skipFrames=skipFrames, skipFramesCnt=skipFramesCnt)
        )
    except Exception as e:
        debug_print(e)
    finally:
        #loop.close()
        loop.run_until_complete(recorder.stop())
        #loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())


