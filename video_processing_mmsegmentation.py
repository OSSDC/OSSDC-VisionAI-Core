import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# MMSegmentation is an open source semantic segmentation toolbox based on PyTorch. It is a part of the OpenMMLab project.
# https://github.com/open-mmlab/mmsegmentation

# Install steps:
# wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes/deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth

# Status: not working


pathToProject='../mmsegmentation/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import torch

from mmdet.apis import inference_detector, init_detector

def init_model(transform):
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path', default='./configs/deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes.py')
    parser.add_argument('checkpoint', help='checkpoint file', default='deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_known_args()

    device = torch.device(args.device)

    model = init_detector(args.config, args.checkpoint, device=device)

    return (device,model),None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (device,model) = processing_model

        result = inference_detector(model, img)

        model.show_result(
            img, result, score_thr=args.score_thr, wait_time=1, show=True)

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("MMDetection Exception",e)
        pass
                
    return tracks,img

