import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. It is a part of the OpenMMLab project.
# https://github.com/open-mmlab/mmdetection

# Install steps:
# wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# wget https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/yolact/yolact_r101_1x8_coco_20200908-4cbe9101.pth
# pip install mmcv-full==1.3.5 terminaltables


# Status: not working


pathToProject='../mmdetection/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import torch

from mmdet.apis import inference_detector, init_detector

def init_model(transform):

    config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    
    
    # config_file = './configs/yolact/yolact_r101_1x8_coco.py'
    # checkpoint_file = './checkpoints/yolact_r101_1x8_coco_20200908-4cbe9101.pth'


    # parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='test config file path', default=config_file)
    parser.add_argument('--checkpoint', help='checkpoint file', default=checkpoint_file)
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args , unknown =  parser.parse_known_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    return (model,args),None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (model,args) = processing_model

        result = inference_detector(model, img)

        model.show_result(
            img, result, score_thr=args.score_thr) #, wait_time=0, show=False)

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("MMDetection Exception",e)
        pass
                
    return tracks,img

