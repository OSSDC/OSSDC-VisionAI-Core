import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# SSD algorithm in Python
# https://github.com/amdegroot/ssd.pytorch

# # Install steps:

# Status: not working


pathToProject='../ssd.pytorch/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

from matplotlib import pyplot as plt
from data import VOC_CLASSES as labels
from data import BaseTransform

count = 0 
def init_model(transform):
    
    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights('./weights/ssd300_mAP_77.43_v2.pth')
    transformer = BaseTransform(net.size, (104, 117, 123))

    return (net,transformer), None

def process_image(transform,processing_model,img):
    global count
    tracks = []
    try:
        count=count+1
        (net,transformer) = processing_model
        if count>0:
            frame = img
            x = cv2.resize(frame, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = x.astype(np.float32)
            x = x[:, :, ::-1].copy()
            # plt.imshow(x)

            x = torch.from_numpy(x).permute(2, 0, 1)     
            xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
            if torch.cuda.is_available():
                xx = xx.cuda()
            y = net(xx)
            detections = y.data

            height, width = frame.shape[:2]
            # scale each detection back up to the image
            scale = torch.Tensor([width, height, width, height])
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                                int(pt[3])), colors[i % 3], 2)
                    cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), font,
                                2, (255, 255, 255), 2, cv2.LINE_AA)
                    j += 1
                
            # top_k=10

            # plt.figure(figsize=(10,10))
            # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            # plt.imshow(rgb_image)  # plot the image for matplotlib
            # currentAxis = plt.gca()

            
            # # scale each detection back up to the image
            # scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
            # for i in range(detections.size(1)):
            #     j = 0
            #     while detections[0,i,j,0] >= 0.6:
            #         score = detections[0,i,j,0]
            #         label_name = labels[i-1]
            #         display_txt = '%s: %.2f'%(label_name, score)
            #         pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            #         coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            #         color = colors[i]
            #         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            #         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            #         j+=1
            # img = predict(img,net,transformer)
            img = frame
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("SSD Pytorch Exception",e)
        pass
                
    return tracks,img

def predict(frame,net,transformer):
    height, width = frame.shape[:2]
    # x = torch.from_numpy(x).permute(2, 0, 1)     
    x = Variable(transformer(frame).unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]),
                          int(pt[3])), colors[i % 3], 2)
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), font,
                        2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame
