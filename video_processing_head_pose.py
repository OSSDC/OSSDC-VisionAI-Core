import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# Head pose estimation
# https://github.com/yinguobing/head-pose-estimation

# Install steps:
# cd ..
# git clone https://github.com/yinguobing/head-pose-estimation

# Status: working

pathToProject='../head-pose-estimation/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

from multiprocessing import Process, Queue
from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# print("OpenCV version: {}".format(cv2.__version__))
    
# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

mark_detector = None
box_process = None
img_queue = None
box_queue = None
pose_estimator = None
pose_stabilizers = None
tm = None
width=-1
height=-1

def init_model(transform):
    global mark_detector,box_process,img_queue,box_queue,pose_estimator,pose_stabilizers, tm
    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()

    # img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()
    return None, None


def process_image(transform,processing_model,img):
    global mark_detector,box_process,img_queue,box_queue,pose_estimator,pose_stabilizers,tm, width, height
    tracks = []
    try:
        frame = img
        h,w,d = frame.shape
        if pose_estimator is None or w!=width or h!=height:
            # sample_frame = frame
            # img_queue.put(sample_frame)

            # Introduce pose estimator to solve pose. Get one frame to setup the
            # estimator according to the image size.
            height, width = h, w
            # (height, width) = (1062 , 485) 
            # (height, width) = (720 , 1280) 
            pose_estimator = PoseEstimator(img_size=(height, width))
        
        # Feed frame to image queue.
        img_queue.put(frame)

        # Get face from box queue.
        facebox = box_queue.get()

        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            tm.start()
            marks = mark_detector.detect_marks(face_img)
            tm.stop()

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

            # Uncomment following line to show facebox.
            # mark_detector.draw_box(frame, [facebox])

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])
            steady_pose = np.reshape(steady_pose, (-1, 3))

            # Uncomment following line to draw pose annotation on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotation on frame.
            pose_estimator.draw_annotation_box(
                frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

            # Uncomment following line to draw head axes on frame.
            # pose_estimator.draw_axes(frame, steady_pose[0], steady_pose[1])
        img = frame
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("HandPose Exception",e)
        pass
                
    return tracks,img

def onClose():
    global box_process
    box_process.terminate()
    box_process.join()


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

