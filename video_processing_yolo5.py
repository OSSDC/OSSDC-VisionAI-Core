import traceback
import argparse
import os
import glob
import random
import time
import cv2

import sys
from datetime import datetime
import os

import numpy as np

import torch
import torch.backends.cudnn as cudnn

pathToProject='../yolo5/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, 
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Object detection using YOLO5 project
# https://github.com/ultralytics/yolov5

# Install steps:
# cd ..
# wget https://github.com/ultralytics/yolov5/archive/v3.1.zip
# unzip v3.1.zip
# ln -s yolov5-3.1 yolo5


# Status: working


def init_model(transform):
    global network, class_names, class_colors,width,height


    device = select_device('')

    # Load model
    # weights = '../yolo5/yolov5m.pt'
    weights = 'yolov5x.pt'
    # weights = 'yolov5l.pt'
    # weights = 'yolov5m.pt'
    # weights = 'yolov5s.pt'
    model = attempt_load(weights, map_location=device)  # load FP32 model

    imgsz = 640
    imgsz = check_img_size(imgsz, s=32)  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img)

    return (device,model,names,colors,img), None


def process_image(transform,processing_model,img):
    global network, class_names, class_colors
    tracks = []
    # imgs = []
    (device,model,names,colors,imgsz) = processing_model
    # view_img = True
    try:
        im0 = img.copy()

        img = letterbox(im0)[0] #, new_shape=(imgsz,imgsz))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=0)#, agnostic=opt.agnostic_nms)

        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            s = '%g: ' % i

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        img = im0
        tracks = pred
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("YOLO 5 Exception",e)
        pass                
    return tracks,img


