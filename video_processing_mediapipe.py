import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# MediaPipe algorithms in Python
# https://google.github.io/mediapipe/getting_started/python.html

# Install steps:
# pip install mediapipe

# Status: working

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic


def init_model(transform):
    if transform == "facemesh":
      face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
      drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
      return (face_mesh,drawing_spec), None
    elif transform == "hands":
      hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
      return (hands), None
    elif transform == "pose":
      pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
      return (pose), None
    elif transform == "holistic":
      holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
      return (holistic), None
    return None,None


def process_image(transform,processing_model,img):
    tracks = []
    try:
      if transform == "facemesh":
        (face_mesh,drawing_spec) = processing_model
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        img = image
        tracks = results.multi_face_landmarks
      elif transform == "hands":
        (hands) = processing_model
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        img = image        
        tracks = results.multi_hand_landmarks
      elif transform == "pose":
        (pose) = processing_model
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        img = image        
        traks = results.pose_landmarks
      elif transform == "holistic":
        (holistic) = processing_model
        image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        img = image        
        # return just face landmarks for now
        traks = results.face_landmarks        
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("MediaPipe Exception",e)
        pass
                
    return tracks,img

