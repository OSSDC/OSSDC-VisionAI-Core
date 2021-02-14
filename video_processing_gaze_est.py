import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# Gaze estimation
# https://github.com/david-wb/gaze-estimation

# Install steps:
# cd ..
# git clone https://github.com/david-wb/gaze-estimation
# cd gaze-estimation; ./scripts/fetch_models.sh

# Status: working

pathToProject='../gaze-estimation/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)


from typing import List, Optional

import torch
from torch.nn import DataParallel

from models.eyenet import EyeNet
import os
import dlib
import imutils
import util.gaze
from imutils import face_utils

from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample

from run_with_webcam import *

torch.backends.cudnn.enabled = True

current_face = None
landmarks = None
alpha = 0.95
left_eye = None
right_eye = None

face_cascade = None
landmarks_detector = None
checkpoint = None
nstack = None
nfeatures = None
nlandmarks = None
eyenet = None


def init_model(transform):
    global face_cascade,landmarks_detector,checkpoint,nstack,nfeatures,nlandmarks,eyenet 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # dirname = os.path.dirname(__file__)
    dirname = pathToProject
    face_cascade = cv2.CascadeClassifier(os.path.join(dirname, 'lbpcascade_frontalface_improved.xml'))
    landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'shape_predictor_5_face_landmarks.dat'))
    # face_cascade = cv2.CascadeClassifier(dirname + 'lbpcascade_frontalface_improved.xml')
    # landmarks_detector = dlib.shape_predictor(dirname +'shape_predictor_5_face_landmarks.dat')

    checkpoint = torch.load('checkpoint.pt', map_location=device)
    # checkpoint = torch.load(dirname + 'checkpoint.pt', map_location=device)
    nstack = checkpoint['nstack']
    nfeatures = checkpoint['nfeatures']
    nlandmarks = checkpoint['nlandmarks']
    eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
    eyenet.load_state_dict(checkpoint['model_state_dict'])
    return None, None


def process_image(transform,processing_model,img):
    global current_face,landmarks,alpha,left_eye,right_eye,face_cascade,landmarks_detector,checkpoint,nstack,nfeatures,nlandmarks,eyenet 

    tracks = []
    try:
        frame_bgr = img
        h,w,d = frame_bgr.shape
        orig_frame = frame_bgr.copy()
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)

        if len(faces):
            next_face = faces[0]
            if current_face is not None:
                current_face = alpha * next_face + (1 - alpha) * current_face
            else:
                current_face = next_face

        if current_face is not None:
            #draw_cascade_face(current_face, orig_frame)
            next_landmarks = detect_landmarks(current_face, gray)

            if landmarks is not None:
                landmarks = next_landmarks * alpha + (1 - alpha) * landmarks
            else:
                landmarks = next_landmarks

            #draw_landmarks(landmarks, orig_frame)


        if landmarks is not None:
            eye_samples = segment_eyes_copy(gray, landmarks)

            eye_preds = run_eyenet(eye_samples)
            left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
            right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

            if left_eyes:
                left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
            if right_eyes:
                right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

            for ep in [left_eye, right_eye]:
                for (x, y) in ep.landmarks[16:33]:
                    color = (0, 255, 0)
                    if ep.eye_sample.is_left:
                        color = (255, 0, 0)
                    cv2.circle(orig_frame,
                               (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)

                gaze = ep.gaze.copy()
                if ep.eye_sample.is_left:
                    gaze[1] = -gaze[1]
                util.gaze.draw_gaze(orig_frame, ep.landmarks[-2], gaze, length=60.0, thickness=2)
        img = orig_frame
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("HandPose Exception",e)
        pass
                
    return tracks,img

def segment_eyes_copy(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)

        if is_left:
            eye_image = np.fliplr(eye_image)
        #     cv2.imshow('left eye image', eye_image)
        # else:
        #     cv2.imshow('right eye image', eye_image)
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes
