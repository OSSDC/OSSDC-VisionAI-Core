import cv2
import numpy as np
import sys
import argparse
# import dlib
from datetime import datetime
import os

# Yolact instance segmentation
# https://github.com/dbolya/yolact
#
# See Google Colab notebook for setup instruction

transform='yolact' #object segmentation

# counter = 0
skip_frames = 30*4

previous_grey = None
hsv = None
hsv_roi = None
roi_hist = None
term_criteria = None
x = 200
y = 350
w = 150
h = 150

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

black = (0, 0, 0)

pathToProject='../yolact/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

from eval_colab import *
from data import config, COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools
from data import cfg, set_cfg, set_dataset

def init_model(transform):
    args = parse_args()

    if args.config is not None:
        print(args.config)
        set_cfg(args.config)
        cfg.mask_proto_debug = False

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')    

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')
        net = net.cuda()

        net = CustomDataParallel(net).cuda()
        transform = torch.nn.DataParallel(FastBaseTransform()).cuda()

    return net, args

def process_image(transform,processing_model,img):
    global previous_grey, hsv, skip_frames,hsv_roi,roi_hist, term_criteria,x, y, w, h
    tracks = []
    try:
        with torch.no_grad():
            net = processing_model
            frame = torch.from_numpy(img).cuda().float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            preds = net(batch)
            # print("display predictions",preds)
            img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

            img = img_numpy
            tracks = preds
    except:
        pass
                
    return tracks,img
