
import traceback
import queue
from pathlib import Path
import cv2

import sys
import argparse
import os
from datetime import datetime, timedelta

import depthai as dai
import numpy as np
from collections import namedtuple
from math import sqrt, ceil
# from FPS import FPS

import time

# OAK-D camera accelerated processing examples
# https://github.com/luxonis/depthai-experiments

# Install steps:
# pip install --extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-snapshot-local/ depthai==0.0.2.1+6ec3f3181b4e46fa6a9f9b20a5b4a3dac5e876b4
# cd ..
# git clone https://github.com/luxonis/depthai-experiments

# implemented these algorithms:
# pre - pedestrian reidentification https://github.com/luxonis/depthai-experiments/tree/master/pedestrian-reidentification
# gaze - gaze estimation https://github.com/luxonis/depthai-experiments/tree/master/gaze-estimation
# age-gen - age gender recognition https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender 

# Status: working

class Anchor:
    def __init__(self, x_center=0, y_center=0, w=0, h=0):
        self.x_center = x_center
        self.y_center = y_center
        self.w = w
        self.h = h

class HandRegion:
    def __init__(self, pd_score, pd_box, pd_kps=0):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor.w = 1.0
                        new_anchor.h = 1.0
                    else:
                        new_anchor.w = anchor_width[anchor_id]
                        new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return anchors

# Create anchors
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt

anchor_options = SSDAnchorOptions(num_layers=4, 
                                    min_scale=0.1484375,
                                    max_scale=0.75,
                                    input_size_height=128,
                                    input_size_width=128,
                                    anchor_offset_x=0.5,
                                    anchor_offset_y=0.5,
                                    strides=[8, 16, 16, 16],
                                    aspect_ratios= [1.0],
                                    reduce_boxes_in_lowest_layer=False,
                                    interpolated_scale_aspect_ratio=1.0,
                                    fixed_anchor_size=True)
anchors = generate_anchors(anchor_options)
print(f"{len(anchors)} anchors have been created")


def decode_bboxes(score_thresh, wi, hi, scores, bboxes, anchors):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    # Decodes the detection tensors generated by the model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }
    """
    sigmoid_scores = 1 / (1 + np.exp(-scores))
    regions = []
    for i,anchor in enumerate(anchors):
        score = sigmoid_scores[i]

        if score > score_thresh:
            # If reverse_output_order is true, sx, sy, w, h = bboxes[i,:4] 
            # Here reverse_output_order is true

            sx, sy, w, h = bboxes[i,:4]
            cx = sx * anchor.w / wi + anchor.x_center 
            cy = sy * anchor.h / hi + anchor.y_center
            w = w * anchor.w / wi
            h = h * anchor.h / hi
            box = [cx - w*0.5, cy - h*0.5, w, h]

            kps = {}
            # 0 : wrist
            # 1 : index finger joint
            # 2 : middle finger joint
            # 3 : ring finger joint
            # 4 : little finger joint
            # 5 : 
            # 6 : thumb joint
            for j, name in enumerate(["0", "1", "2", "3", "4", "5", "6"]):
                # Here reverse_output_order is true
                lx, ly = bboxes[i,4+j*2:6+j*2]
                lx = lx * anchor.w / wi + anchor.x_center 
                ly = ly * anchor.h / hi + anchor.y_center
                kps[name] = [lx, ly]
            regions.append(HandRegion(float(score), box, kps))
    return regions

def non_max_suppression(regions, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in regions]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
    scores = [r.pd_score for r in regions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i[0]] for i in indices]



frame = None
bboxes = []

# fps = FPS()

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height


def frame_norm3(frame, bbox):
    # frame.shape[:2],
    fh, fw, d = frame.shape
    sw = fw // 128
    sh = fh // 128
    x = bbox[0] * sw 
    y = bbox[1] * sh
    w = bbox[2] * sw
    h = bbox[3] * sh
    return [x,y,w,h]


pd_score_thresh = 0.5
pd_nms_thresh = 0.3

device = None
q_rgb = None
q_nn = None
q_in = None

debug=True

useOAKDCam=False

def init_model(transform):

    if transform == 'oakd_palm':
        # # Start defining a pipeline
        # pipeline = dai.Pipeline()

        # if useOAKDCam:
        #     # Define a source - color camera
        #     cam_rgb = pipeline.createColorCamera()
        #     cam_rgb.setPreviewSize(128, 128)
        #     cam_rgb.setFps(90.0)
        #     cam_rgb.setInterleaved(False)


        # # Define a neural network that will make predictions based on the source frames
        # detection_nn = pipeline.createNeuralNetwork()
        # detection_nn.setBlobPath(str(Path("../oakd_palm_detection/models/palm_detection.blob").resolve().absolute()))

        # cam_rgb.preview.link(detection_nn.input)

        # # Create outputs
        # xout_rgb = pipeline.createXLinkOut()
        # xout_rgb.setStreamName("rgb")
        # cam_rgb.preview.link(xout_rgb.input)

        # xout_nn = pipeline.createXLinkOut()
        # xout_nn.setStreamName("nn")
        # detection_nn.out.link(xout_nn.input)

        # # Pipeline defined, now the device is assigned and pipeline is started
        # device = dai.Device(pipeline)
        # device.startPipeline()

        # if useOAKDCam:
        #     # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        #     q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        #     q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # return (q_rgb,q_nn), None
        return None, None

def process_image(transform,processing_model,img):
    global useOAKDCam, bboxes, results, pd_score_thresh, pd_nms_thresh, bboxes, anchors, device, q_rgb, q_nn, fps, q_in
    tracks = []
    # (q_rgb,q_nn) = processing_model
    try:
        # if useOAKDCam:
        # #     ret, frame = cap.read()
        #     frame = np.array(cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)        
        #     shape = (3, frame.getHeight(), frame.getWidth())
        #     frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        #     frame = np.ascontiguousarray(frame)
        # else:
        frame = img

        #palm detection https://github.com/geaxgx/oakd_palm_detection
        if transform == 'oakd_palm':
            if device is None:
                    # Start defining a pipeline
                    pipeline = dai.Pipeline()

                    if useOAKDCam:
                        # Define a source - color camera
                        cam_rgb = pipeline.createColorCamera()
                        cam_rgb.setPreviewSize(128, 128)
                        cam_rgb.setFps(90.0)
                        cam_rgb.setInterleaved(False)

                    # Define a neural network that will make predictions based on the source frames
                    detection_nn = pipeline.createNeuralNetwork()
                    detection_nn.setBlobPath(str(Path("../oakd_palm_detection/models/palm_detection.blob").resolve().absolute()))

                    if useOAKDCam:
                        cam_rgb.preview.link(detection_nn.input)
                    else:
                        detection_in = pipeline.createXLinkIn()
                        detection_in.setStreamName("detection_in")
                        detection_in.out.link(detection_nn.input)

                    # Create outputs
                    if useOAKDCam:
                        xout_rgb = pipeline.createXLinkOut()
                        xout_rgb.setStreamName("rgb")
                        cam_rgb.preview.link(xout_rgb.input)

                    xout_nn = pipeline.createXLinkOut()
                    xout_nn.setStreamName("nn")
                    detection_nn.out.link(xout_nn.input)

                    # Pipeline defined, now the device is assigned and pipeline is started
                    device = dai.Device(pipeline)
                    device.startPipeline()

                    if useOAKDCam:
                        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
                        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                    else:
                        q_in = device.getInputQueue("detection_in")
                        
                    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            # fps.update()
            # if frame is not None:

            if not useOAKDCam:
                nn_data = dai.NNData()
                nn_data.setLayer("input", to_planar(frame, (128, 128)))
                q_in.send(nn_data)

                # in_nn = q_nn.get()
                in_nn = q_nn.tryGet()
                # 2 output layers:
                # - classificators:
                # - regressors : 
                # From: print(in_nn.getAllLayerNames())

                if in_nn is not None:
                    scores = np.array(in_nn.getLayerFp16("classificators"))
                    bboxes = np.array(in_nn.getLayerFp16("regressors")).reshape((896,18))

                    # Decode bboxes
                    regions = decode_bboxes(pd_score_thresh, 128, 128, scores, bboxes, anchors)
                    # Non maximum suppression
                    regions = non_max_suppression(regions, pd_nms_thresh)
                    tracks = regions
                    for r in regions:
                        raw_bbox = (np.array(r.pd_box) * 128).astype(int)
                        # box = raw_bbox
                        # print("raw_bbox",raw_bbox)
                        # print("frame.shape",frame.shape)
                        box = frame_norm3(frame, raw_bbox)
                        # print("box3",box)
                        # cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
                        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255,0), 2)
                        
                    # if frame is not None:
                    #     img = frame
                    if frame is not None:
                        # cv2.putText(frame, "FPS: {:.2f}".format(fps.get()), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                        # cv2.imshow("rgb", frame)
                        img = frame

            else:
                # in_rgb = q_rgb.tryGet()

                in_rgb = q_rgb.get()
                
                if in_rgb is not None:
                    # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
                    shape = (3, in_rgb.getHeight(), in_rgb.getWidth())

                    frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                    frame = np.ascontiguousarray(frame)
                    in_nn = q_nn.get()
                    # 2 output layers:
                    # - classificators:
                    # - regressors : 
                    # From: print(in_nn.getAllLayerNames())

                    if in_nn is not None:
                        scores = np.array(in_nn.getLayerFp16("classificators"))
                        bboxes = np.array(in_nn.getLayerFp16("regressors")).reshape((896,18))

                        # Decode bboxes
                        regions = decode_bboxes(pd_score_thresh, 128, 128, scores, bboxes, anchors)
                        # Non maximum suppression
                        regions = non_max_suppression(regions, pd_nms_thresh)
                        tracks = regions
                        for r in regions:
                            box = (np.array(r.pd_box) * 128).astype(int)
                            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,255,0), 2)
                        
                    # if frame is not None:
                    #     img = frame
                    if frame is not None:
                        # cv2.putText(frame, "FPS: {:.2f}".format(fps.get()), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
                        # cv2.imshow("rgb", frame)
                        img = frame

                    # if cv2.waitKey(1) == ord('q'):
                    #     pass


    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("OAK-D Exception",e)
        pass
                
    return tracks,img

def create_pipeline_palm():
    global useOAKDCam
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if useOAKDCam:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(300, 300)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # Define a neural network that will make predictions based on the source frames
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("../oakd_palm_detection/models/palm_detection.blob").resolve().absolute()))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)

    if useOAKDCam:
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    print("Pipeline created.")
    return pipeline
