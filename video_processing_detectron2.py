import traceback
import cv2
import numpy as np
import sys
import argparse
from datetime import datetime
import os

# Detectron2 algorithm in Python
# https://github.com/facebookresearch/detectron2

# # Install steps:
# 
# install detectron2
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

# Status: working

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def init_model(transform):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return (predictor,cfg), None

def process_image(transform,processing_model,img):
    tracks = []
    (predictor,cfg) = processing_model
    try:
        outputs = predictor(img)
        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        # print(outputs["instances"].pred_classes)
        # print(outputs["instances"].pred_boxes)
        tracks = outputs["instances"].pred_boxes
        # We can use `Visualizer` to draw the predictions on the image.
        vis = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        img = out.get_image()[:, :, ::-1]
    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("MediaPipe Exception",e)
        pass
                
    return tracks,img

