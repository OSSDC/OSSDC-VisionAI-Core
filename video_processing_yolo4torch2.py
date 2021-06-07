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

# This is PyTorch implementation of YOLOv4 which is based on ultralytics/yolov3.
# https://github.com/WongKinYiu/PyTorch_YOLOv4


# Status: not working

pathToProject='../PyTorch_YOLOv4/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from utils.torch_utils import select_device, time_synchronized

from models.models import *


def init_model(transform):
    global network, class_names, class_colors,width,height

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-pacsp-x-mish_.weights', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--merge', action='store_true', help='use Merge NMS')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--cfg', type=str, default='./cfg/yolov4.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./data/coco.names', help='*.cfg path')
    opt, unknown = parser.parse_known_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    res = prepareModel(opt, opt.data,
            [opt.weights],
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls,
            opt.augment,
            opt.verbose)


    return res, None


def process_image(transform,processing_model,img):
    global network, class_names, class_colors
    tracks = []
    # imgs = []
    (device,model,names,colors,imgsz) = processing_model
    # view_img = True
    try:
        # im0 = img.copy()


        # img = im0
        img = process_image(img,processing_model)
        tracks = pred
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("YOLO4 PyTorch 2 Exception",e)
        pass                
    return tracks,img


def prepareModel(opt, data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir='',
         merge=False,
         save_txt=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(opt.device, batch_size=batch_size)
        merge, save_txt = opt.merge, opt.save_txt  # use Merge NMS, save *.txt labels
        if save_txt:
            out = Path('inference/output')
            if os.path.exists(out):
                shutil.rmtree(out)  # delete output folder
            os.makedirs(out)  # make new output folder

        # Remove previous
        for f in glob.glob(str(Path(save_dir) / 'test_batch*.jpg')):
            os.remove(f)

        # Load model
        model = Darknet(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=32)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 32, opt,
                                       hyp=None, augment=False, cache=False, pad=0.5, rect=True)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(opt.names)
    coco91class = coco80_to_coco91_class()
    return (model,names)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def process_image(img, processing_model):

    (model,names) = processing_model

    # Disable gradients
    with torch.no_grad():
        # Run model
        t = time_synchronized()
        inf_out, train_out = model(img, augment=augment)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if training:  # if model has loss hyperparameters
            loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # GIoU, obj, cls

        # Run NMS
        t = time_synchronized()
        output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
        t1 += time_synchronized() - t

    # Statistics per image
    for si, pred in enumerate(output):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        seen += 1

        if pred is None:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Append to text file
        if save_txt:
            gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
            txt_path = str(out / Path(paths[si]).stem)
            pred[:, :4] = scale_coords(img[si].shape[1:], pred[:, :4], shapes[si][0], shapes[si][1])  # to original
            for *xyxy, conf, cls in pred:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        # Append to pycocotools JSON dictionary
        if save_json:
            # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            image_id = Path(paths[si]).stem
            box = pred[:, :4].clone()  # xyxy
            scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
            box = xyxy2xywh(box)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                jdict.append({'image_id': int(image_id) if image_id.isnumeric() else image_id,
                                'category_id': coco91class[int(p[5])],
                                'bbox': [round(x, 3) for x in b],
                                'score': round(p[4], 5)})

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5]) * whwh

            # Per target class
            for cls in torch.unique(tcls_tensor):
                ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                    # Append detections
                    for j in (ious > iouv[0]).nonzero(as_tuple=False):
                        d = ti[i[j]]  # detected target
                        if d not in detected:
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Plot images
    if batch_i < 1:
        f = Path(save_dir) / ('test_batch%g_gt.jpg' % batch_i)  # filename
        plot_images(img, targets, paths, str(f), names)  # ground truth
        f = Path(save_dir) / ('test_batch%g_pred.jpg' % batch_i)
        plot_images(img, output_to_target(output, width, height), paths, str(f), names)  # predictions

    return img