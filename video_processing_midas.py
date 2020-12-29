import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# Mono depth using MiDaS project - old model
# https://github.com/intel-isl/MiDaS

# Install steps:
# pip install pytorch torchvision
# cd ..
# git clone https://github.com/intel-isl/MiDaS
# wget https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt MiDaS/model-f46da743.pt

# Status: working

sys.path.insert(0, '../MiDaS/')

import torch
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

    
def init_model(transform):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model_path = "../MiDaS/model-f46da743.pt"

    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    model = MidasNet(model_path, non_negative=True)

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()
    return (model, transform, device), None


def process_image(transform,processing_model,img):
    global previous_grey, hsv, skip_frames,hsv_roi,roi_hist, term_criteria,x, y, w, h
    tracks = []
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = get_depth(img,processing_model[0],processing_model[1], processing_model[2])
        img = (img/256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
#         return img
    except:
        pass
                
    return tracks,img


def depth_to_image(depth, bits=1):
    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        img = out.astype("uint8")
    elif bits == 2:
        img = out.astype("uint16")

    return img

def get_depth(img, model,transform,device):
    
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    img = depth_to_image(prediction, bits=2)
    return img
