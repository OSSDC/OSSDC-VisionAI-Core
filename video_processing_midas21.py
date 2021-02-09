import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# Mono depth using MiDaS project - new 2.1 model
# https://github.com/intel-isl/MiDaS

# Status: working

sys.path.insert(0, '../MiDaS/')

import torch
from torchvision.transforms import Compose
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet

    
def init_model(transform):
    parser = argparse.ArgumentParser()
    parser.add_argument('-mw', '--model_weights', 
        default='model-f6b98070.pt',
        help='path to the trained weights of model'
    )

    parser.add_argument('-mt', '--model_type', 
        default='large',
        help='model type: large or small'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

    args, unknown = parser.parse_known_args()    
    
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if args.model_type == "large":
        model_path = "../MiDaS/"+args.model_weights
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
    elif args.model_type == "small":
        if "small" not in args.model_weights:
            args.model_weights = "model-small-70d6b9c8.pt"
        model_path = "../MiDaS/"+args.model_weights
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
        net_w, net_h = 256, 256
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
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

    model.eval()
    
    if args.optimize==True:
        rand_example = torch.rand(1, 3, net_h, net_w)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

    model.to(device)    
    
    return (model, transform, device, args.optimize), args


def process_image(transform,processing_model,img):
    global previous_grey, hsv, skip_frames,hsv_roi,roi_hist, term_criteria,x, y, w, h
    tracks = []
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img = get_depth(img,processing_model[0],processing_model[1], processing_model[2], processing_model[3])
        img = (img/256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("OAK-D Exception",e)
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

def get_depth(img, model,transform,device,optimize):
    
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize==True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
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

