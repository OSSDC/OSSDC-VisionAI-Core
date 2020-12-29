import asyncio
import json
import logging
import os
import random
import sys

from aiortc import RTCIceCandidate, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp

import socketio
import time
import hashlib

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None

logger = logging.getLogger("signaling_race")
BYE = object()

start_timer = None

roomName = 'TestRoom12345' #replace this with your room name and password
passCode = '12345'
droneRoomName = hashlib.md5((roomName+'-'+passCode).encode("utf-8")).hexdigest()

debug=False

def debug_print(*argv):
    if(debug):
        debug_print(*argv)
    

sio = socketio.AsyncClient()

sio_messages = []

@sio.on(droneRoomName+'.members',namespace='/')
async def on_room_members(data):
    print("on_room_members",data)
    sio_messages.append(data)

@sio.on(droneRoomName+'.message',namespace='/')
async def on_room_message(data):
    if data["sender"] == sio.eio.sid:
        return
    debug_print("on_room_message",data)
    sio_messages.append(data)

@sio.on('ping',namespace='/')
async def onping_message(data):
    debug_print("onping_message",data)
    sendMessage("{'pong': true}")

@sio.on
async def on_message(sid,data):
    if sid==sio.eio.sid:
        return
    debug_print("on_message",data)
    sio_messages.append(data)

@sio.event
async def connect():
    debug_print('connection established')
    debug_print("sid", sio.eio.sid)

async def sendMessage(message):
    message = {"roomName": droneRoomName, "message": message}
    debug_print("sendMessage msg",message)
    res = await sio.emit('publish', message)
    debug_print("sendMessage res",res)

async def sendSubscribeMessage():
    debug_print("sendSubscribeMessage", droneRoomName)
    await sio.emit('subscribe', droneRoomName)
    
async def sendUnSubscribeMessage():
    debug_print("sendUnSubscribeMessage", droneRoomName)
    await sio.emit('unsubscribe', droneRoomName)

async def object_from_string(message_str):
    if(isinstance(message_str,list)):
        return message_str
    message = None;
    if(isinstance(message_str,dict)):
        if "sdp" in message_str["message"]:
            message = message_str["message"]["sdp"]
        if "candidate" in message_str["message"]:
            message = message_str["message"] #["candidate"]
        if "stick" in message_str["message"]:
            message = message_str["message"]
    else:
        message = json.loads(message_str)

    if message is not None:
        if "type" in message and message["type"] in ["answer", "offer"]:
            return RTCSessionDescription(**message)
        elif ("type" in message and message["type"] == "candidate" and message["candidate"]) or message["candidate"]:
            candidate = None
            if(isinstance(message["candidate"], dict )):
                candidate = candidate_from_sdp(message["candidate"]["candidate"].split(":", 1)[1])
            else:
                candidate = candidate_from_sdp(message["candidate"].split(":", 1)[1])
            if(isinstance(message["candidate"], dict )):
                candidate.sdpMid = message["candidate"]["sdpMid"]
                candidate.sdpMLineIndex = message["candidate"]["sdpMLineIndex"]
            else:
                candidate.sdpMid = message["id"]
                candidate.sdpMLineIndex = message["label"]
            return candidate
        elif message["type"] == "bye":
            return BYE
            
    return message


async def object_to_string(obj):
    if isinstance(obj, RTCSessionDescription):
        message = {"sdp": {"sdp": obj.sdp, "type": obj.type}}
    elif isinstance(obj, RTCIceCandidate):
        if hasattr(obj, 'label'):
            message = {
                "candidate": "candidate:" + candidate_to_sdp(obj),
                "id": obj.id, #obj.sdpMid,
                "label": obj.label, #obj.sdpMLineIndex,
                "type": "candidate",
            }
        else:
            message = {
                "candidate": "candidate:" + candidate_to_sdp(obj),
                "id": obj.sdpMid,
                "label": obj.sdpMLineIndex,
                "type": "candidate",
            }

    else:
        return obj
    return message


class RaceOssdcSignaling:
    def __init__(self, room):
        self._http = None
        self._origin = "https://race.ossdc.org"
        self._room = room
        self._room = roomName
        self.trackEnded = False

    async def connect(self):
        join_url = self._origin + "/join/#" + self._room

        await sio.connect(self._origin)
        params = {}
        params["is_initiator"] = "true"

        self.__is_initiator = params["is_initiator"] == "true"

        return params

    async def receive(self):
        global sio_messages
        
        while(not sio_messages):
            if self.trackEnded:
              return None
            await sio.sleep(1)
        if sio_messages:
            message = sio_messages.pop(0)
        debug_print("receive",message)
        return await object_from_string(message)

    async def send(self, obj):
        message = await object_to_string(obj)
        await sendMessage(message)
