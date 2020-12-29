import cv2
import numpy as np
import sys
import dlib
from datetime import datetime
import os

# DLIB based face landmarks detection

# Status: working

# Install steps:
# pip install dlib
# cd ..
# # See this for lfs install https://docs.github.com/en/free-pro-team@latest/github/managing-large-files/installing-git-large-file-storage
# git lfs install --skip-repo
# git clone https://github.com/OSSDC/OSSDC-VisionAI-Datasets/

def init_model(transform):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../OSSDC-VisionAI-Datasets/pretrained/face_landmarks/shape_predictor_68_face_landmarks.dat")
    return (detector,predictor), None


def process_image(transform,processing_model,img):
    global previous_grey, hsv, skip_frames,hsv_roi,roi_hist, term_criteria,x, y, w, h
    tracks = []
    try:
      (detector,predictor) = processing_model

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      faces = detector(gray)
      for face in faces:
          x1 = face.left()
          y1 = face.top()
          x2 = face.right()
          y2 = face.bottom()
          #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

          landmarks = predictor(gray, face)

          for n in range(0, 68):
              x = landmarks.part(n).x
              y = landmarks.part(n).y
              cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

    except:
        pass
                
    return tracks,img

