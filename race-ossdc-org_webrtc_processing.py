import argparse
import asyncio
import logging
import os
import random
import numpy as np
import cv2
from av import VideoFrame
import traceback 

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
from signaling_race import BYE, RaceOssdcSignaling, object_from_string, object_to_string, sendSubscribeMessage,sendUnSubscribeMessage, roomName, sendMessage,droneRoomName,sio

import sys
import argparse

import time
import json

import socket

import youtube_dl

import subprocess

debug=True

def debug_print(*argv):
    if(debug):
        print(*argv)

#To monitor the output from video processing run on your PC this command:
#ffplay -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
ip = 'localhost' #replace with your PC IP where ffplay runs
ip = None #comment to activate above IP

clientsocket = None
if ip is not None:
  try:
    clientsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    clientsocket.settimeout(5)
    clientsocket.connect((ip,45654)) #the target ip is where the ffplay is listening
  except Exception as e:
    debug_print(e)
    clientsocket = None


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, signaling, model):
        super().__init__()  # don't forget this!
        self.transform = transform
        self.track = track
        self.scale = 1
        self.skipFrames = True
#         self.skipFrames = False #use it for Youtube streaming
        self.processing_model = model
        self.prevTime = time.time()
        self.starttime1 = time.time()
        self.colors = "Not computed"
        self.signaling = signaling
        self.frameProcessedCount = 0
        self.frameCount = 0
        self.prevFrameCount = 0
        self.realFPS = 0
        
    async def recv(self):
        global clientsocket
        frame = await self.track.recv()
        self.frameCount=self.frameCount+1
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
            trakingPoints,img = video_processing_module.process_image(self.transform,self.processing_model,img)

            y1 = y+25

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

                    cv2.putText(img, self.colors[0], (10,y1+125+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv2.putText(img, self.colors[1], (10,y1+150+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv2.putText(img, self.colors[2], (10,y1+175+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    #         img = cv2.resize(img,(cols//4, rows//4))
    #         y = h//3
    #         x = w//3
    #         y1 = y+25                

            cv2.putText(img, "ImgSize: "+str(w)+"x"+str(h), (10,y1+50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(img, "FrmCnt: "+str(self.frameCount), (10,y1+75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(img, "FrmProcCnt: "+str(self.frameProcessedCount), (10,y1+100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(img, "TrkPt: "+str(len(trakingPoints)), (10,y1+125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            cv2.putText(img, "ProcFPS : " + str(int(fps)), (10,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(img, "RealFPS : " + str(int(self.realFPS)), (10,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    #         img = cv2.resize(img,(cols//3, rows//3))

            delta = time.time() - self.prevTime
            if delta > 1:
                 self.realFPS = (self.frameCount-self.prevFrameCount)/delta
                 self.prevFrameCount = self.frameCount
                 self.prevTime = time.time()


            try:
                if clientsocket is not None:
                    #img = img.to_ndarray(format="bgr24")
                    data = cv2.imencode('.jpg', img)[1].tobytes()
                    clientsocket.send(data)          
            except Exception as e:
                debug_print(e)
                clientsocket = None
                pass     

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base 
        except Exception as e1:
            debug_print(e1)

        return new_frame


async def run(pc, player, recorder, signaling, transform, model):
    def add_tracks():
      debug_print("player",player)
      if player and player.video:
          local_video = player.video
          local_video = VideoTransformTrack(player.video, transform=transform, signaling=signaling, model=model)
          pc.addTrack(local_video)
      else:
          pc.addTransceiver('video','sendrecv') #this is the trick to echo webcam back

    @pc.on("track")
    def on_track(track):
      debug_print("Track %s received" % track.kind)
      if track.kind == "video":
        if not(player and player.video):
            local_video = VideoTransformTrack(
                track, transform=transform, signaling=signaling, model=model
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
          debug_print("track ended")
          signaling.trackEnded=True

    trackEnded = False
    params = await signaling.connect()

    await sendSubscribeMessage()

    debug_print("run")
    # consume signaling
    noneCnt = 0
    while True:
        obj = await signaling.receive()
#         debug_print("obj:", obj)
        if obj is None:
            if(noneCnt>5):
                break
            noneCnt=noneCnt+1
            continue
        try:
          if hasattr(obj, 'type'):
              if obj.type == "answer":
                  if pc.signalingState == "stable":
                      pass
                  else:
                      # add_tracks()
                      await pc.setRemoteDescription(obj)
                      await recorder.start()
                      await signaling.send(pc.localDescription)
              if obj.type == "offer":
                  if pc.signalingState == "have-local-offer" or pc.signalingState == "stable":
                      pass
                  else:
                      # add_tracks()
                      await pc.setRemoteDescription(obj)
                      await recorder.start()
                      await signaling.send(pc.localDescription)

          if(isinstance(obj,list) and len(obj)==2):
              add_tracks()
              await pc.setLocalDescription(await pc.createOffer())
              await signaling.send(pc.localDescription)
          elif isinstance(obj, RTCSessionDescription):
              debug_print("pc.signalingState",pc.signalingState)

          elif isinstance(obj, RTCIceCandidate):
                await pc.addIceCandidate(obj)

          elif obj is BYE or signaling.trackEnded:
                debug_print("Exiting")
                break
          else:
              debug_print("obj not handled:",obj)
        except Exception as e:
            noneCnt=noneCnt+1
            debug_print("error in run loop",e)
            if(noneCnt>5):
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RaceOSSDC")
    
    parser.add_argument("--play-from", help="Read the media from a file and sent it."),
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument('-t','--transform', type=str)
    args, unknown = parser.parse_known_args()

    transform = None
    if args.transform:
        transform = args.transform

    import importlib

    module_name = transform.split(".")
    if len(module_name) == 2:
           transform = module_name[1]
           module_name = module_name[0]
    else:
           module_name = module_name[0]
           transform = module_name

    debug_print("We will apply this transform:",transform, "from module:",module_name)
        
    video_processing_module = importlib.import_module("video_processing_"+module_name)
    print('video_processing_module',video_processing_module)
    
    
    model,args1 = video_processing_module.init_model(transform)
    
    # create signaling and peer connection
    args.room = '123456'

    videoUrl = None 
#     videoUrl = 'https://youtu.be/uuQlMCMT71I' #uncomment to overide with a Youtube video source, set skipFrames to False for Youtube streaming
  
    if videoUrl is not None:
        #install youtube-dl for this to work: pip install youtube-dl
        command = "youtube-dl -f 'bestvideo[height<1100]' -g '"+videoUrl+"'" 
        videoUrl = subprocess.check_output(command, shell = True).decode("utf-8").strip()
        args.play_from = videoUrl

    print('videoUrl=',videoUrl)

    signaling = RaceOssdcSignaling(args.room)
    
    configuration = RTCConfiguration()
    
    stunServer = RTCIceServer("stun:race.ossdc.org:5349")

    configuration.iceServers = []
    configuration.iceServers.append(stunServer)

    pc = RTCPeerConnection(configuration=configuration)


    # create media source
    if args.play_from:
        player = MediaPlayer(args.play_from)
    else:
        player = None

    # create media sink
    if args.record_to:
        recorder = MediaRecorder(args.record_to)
    else:
        recorder = MediaBlackhole()

    loop = asyncio.get_event_loop()

    try:
        loop.run_until_complete(
            run(pc=pc, player=player, recorder=recorder, signaling=signaling,transform=transform, model=model)
        )
    except Exception as e:
        debug_print(e)
    finally:
        loop.close()
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())

