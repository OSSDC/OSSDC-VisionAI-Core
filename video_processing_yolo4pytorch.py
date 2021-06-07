from __future__ import division

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


# Object detection using YOLO4 project
# https://github.com/Tianxiaomo/pytorch-YOLOv4

# Install steps:
# mkdir wights
# gdown https://drive.google.com/file/d/1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ -O weights/yolov4.pth
# gdown https://drive.google.com/file/d/1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA -O weights/yolov4.conv.137.pth

# Status: nor working

pathToProject='../pytorch-YOLOv4/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

from tool.darknet2pytorch import Darknet
from tool.utils import *
from tool.torch_utils import *

def init_model(transform):

    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.25)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="160", type=str)
    args, unknown = parser.parse_known_args()

    cfgfile = "./cfg/yolov4.cfg"
    weightsfile = "./weights/yolov4.pth"

    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    CUDA = torch.cuda.is_available()
    num_classes = 80
    # bbox_attrs = 5 + num_classes
    class_names = load_class_names("./data/coco.names")

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)

    if CUDA:
        model.cuda()

    model.eval()
    return (model, class_names,CUDA), None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (model, class_names, CUDA) = processing_model
        sized = cv2.resize(img, (model.width, model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(model, sized, 0.5, 0.4, CUDA)

        if len(boxes)>1
            img = plot_boxes_cv2(img, boxes, class_names=class_names)
            tracks = boxes
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("YOLO 4 PyTorch Exception",e)
        pass                
    return tracks,img


