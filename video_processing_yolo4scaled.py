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


# Scaled YOLOv4
# https://github.com/DataXujing/ScaledYOLOv4

# Install steps:
# mkdir wights
# cd weights/
# gdown https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz -O yolov4-p5.pt
# gdown https://drive.google.com/file/d/1aB7May8oPYzBqbgwYSZHuATPXyxh9xnf -O yolov4-p6.pt
# gdown https://drive.google.com/file/d/18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3 -O yolov4-p7.pt

# Status: nor working

pathToProject='../ScaledYOLOv4/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import shutil

import yaml

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def init_model(transform):


    input_size = 640

    weights = "./weights/yolov4-p5.pt"   # 
    weights = "./weights/yolov4-p6.pt"   # 
    weights = "./weights/yolov4-p7.pt"   # 

    device = select_device("0", batch_size=1)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(input_size, s=model.stride.max())  # check img_size
    
    # bbox_attrs = 5 + num_classes
    # class_names = load_class_names("./data/coco.names")

    single_cls = False
    with open('./data/coco.yaml') as f:
        class_names = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(class_names['nc'])  # number of classes

    # config
    model.eval()
    return (model, class_names['names'], input_size,device), None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (model, names, input_size,device) = processing_model
    
        prob_thres = 0.2        # NMS, 0.3, 0.5, 0.6, 0.7,0.75, 0.8, 0.85, 0.9
        conf_thres = 0.4       # NMS
        iou_thres = 0.5          # NMS
        merge = True             # NMS oxes merged using weighted mean

        # img0 = cv2.imread(img_path)
        img0 = img.copy()
        # Padded resize
        img = letterbox(img0, new_shape=input_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)

        # torch tensor
        img = torch.from_numpy(img).to(device)
        img = img.float()
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # nb, _, height, width = img.shape  # batch size, channels, height, width
        # whwh = torch.Tensor([width, height, width, height]).to(device)

        # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=True)[0]   #TTA

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
        # t2 = time_synchronized()

        # Process detections

        det_count = 0

        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls_ in det:   # x1,y1,x2,y2
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # if save_img or view_img:  # Add bbox to image
                        # label = '%s' % (names[int(cls)])
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)


                    label = '%s' % (names[int(cls_)])

                    # if not os.path.exists("./metric/detections"):
                    #     os.makedirs("./metric/detections")

                    # if label in ["A","B","C","D","E","N1","N2","N3","N4","N5","N6","N7","N8","N9","N10"]:
                    #     continue
                    # if conf <= prob_thres:
                    #     continue

                    if label not in ["car", "truck"]:
                        continue

                    # det_count += 1

                    label_text = label #names2label[label]
                    # print(conf.cpu().detach().numpy())
                    prob = round(conf.cpu().detach().numpy().item(),2)


                    # tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

                    color = (255, 255, 0)
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                    cv2.rectangle(img0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label_text+":"+str(prob), 0, fontScale=tl / 1.5, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.rectangle(img0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img0, label_text+":"+str(prob), (c1[0], c1[1] - 2), 0, tl / 1.5, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

        img = img0

        tracks = pred
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("YOLO 4 Scaled PyTorch  Exception",e)
        pass                
    return tracks,img


