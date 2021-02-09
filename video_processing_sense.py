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

pathToProject='../sense/'
sys.path.insert(0, pathToProject)
# os.chdir(pathToProject)

# Activity recognition
# https://github.com/TwentyBN/sense

# Install steps:
# cd ..
# git clone https://github.com/TwentyBN/sense
# # Create an account and download pretrained models https://20bn.com/licensing/sdk/evaluation
# # unzip the pretrained models in sense folder

# Status: 
#  - gesture - working
#  - fittnes - not working

# import sense
import sense.display
from sense import feature_extractors
from sense.controller import Controller
from sense.downstream_tasks.gesture_recognition import INT2LAB
from sense.downstream_tasks.gesture_recognition import LAB_THRESHOLDS
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.nn_utils import load_weights_from_resources
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.engine import InferenceEngine
from typing import Optional
from typing import Tuple
from sense.downstream_tasks import calorie_estimation


frame_index = None
clip = None

def init_model(transform):
    use_gpu = True
    inference_engine = None
    neural_network = None
    postprocessors = None

    if transform == 'gesture':
        # Load feature extractor
        feature_extractor = feature_extractors.StridedInflatedEfficientNet()
        feature_extractor.load_weights_from_resources('backbone/strided_inflated_efficientnet.ckpt')
        feature_extractor.eval()

        # Load a logistic regression classifier
        gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                                num_out=30)
        checkpoint = load_weights_from_resources('gesture_detection/efficientnet_logistic_regression.ckpt')
        gesture_classifier.load_state_dict(checkpoint)
        gesture_classifier.eval()

        # Concatenate feature extractor and met converter
        neural_network = Pipe(feature_extractor, gesture_classifier)
        postprocessors = [ PostprocessClassificationOutput(INT2LAB, smoothing=4) ]

    elif transform == 'fitness':
        weight = float(60)
        height = float(170)
        age = float(20)
        gender = 'female'
    
        # Load feature extractor
        feature_extractor = feature_extractors.StridedInflatedMobileNetV2()
        feature_extractor.load_weights_from_resources('backbone/strided_inflated_mobilenet.ckpt')
        feature_extractor.eval()

        # Load fitness activity classifier
        gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                                num_out=81)
        checkpoint = load_weights_from_resources('fitness_activity_recognition/mobilenet_logistic_regression.ckpt')
        gesture_classifier.load_state_dict(checkpoint)
        gesture_classifier.eval()

        # Load MET value converter
        met_value_converter = calorie_estimation.METValueMLPConverter()
        checkpoint = load_weights_from_resources('calorie_estimation/mobilenet_features_met_converter.ckpt')
        met_value_converter.load_state_dict(checkpoint)
        met_value_converter.eval()    


        # Concatenate feature extractor with downstream nets
        neural_network = Pipe(feature_extractor, feature_converter=[gesture_classifier,
                                                        met_value_converter])

        postprocessors = [
            PostprocessClassificationOutput(INT2LAB, smoothing=8,
                                            indices=[0]),
            calorie_estimation.CalorieAccumulator(weight=weight,
                                                height=height,
                                                age=age,
                                                gender=gender,
                                                smoothing=12,
                                                indices=[1])
        ]

    if neural_network is not None:
        inference_engine = InferenceEngine(neural_network, use_gpu=use_gpu)
        start_inference(inference_engine)

    return (inference_engine,postprocessors), None


def process_image(transform,processing_model,img):
    global clip, frame_index
    tracks = []
    (inference_engine,postprocessors) = processing_model

    try:
        imgBack = img

        if inference_engine is not None and (transform == 'gesture' or transform == 'fitness'):

            frame_index += 1

            img_tuple = get_image(img,inference_engine)

            # # If not possible, stop
            # if img_tuple is None:
            #     break

            # Unpack
            img, numpy_img = img_tuple
            
            clip = np.roll(clip, -1, 1)
            clip[:, -1, :, :, :] = numpy_img

            if frame_index == inference_engine.step_size:
                # A new clip is ready
                inference_engine.put_nowait(clip)

            frame_index = frame_index % inference_engine.step_size

            # Get predictions
            prediction = inference_engine.get_nowait()

            if postprocessors is not None:
                prediction_postprocessed = postprocess_prediction(postprocessors,prediction)
            else:
                prediction_postprocessed = prediction

            # print("prediction_postprocessed:",prediction_postprocessed)
            # controller.display_prediction(img, prediction_postprocessed)

            if prediction_postprocessed is not None:
                (label, pred) = prediction_postprocessed['sorted_predictions'][0]
                # print(label, pred)
                cv2.putText(imgBack, "Pred: "+label + " "+"{:.2f}".format(pred), (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

        img = imgBack

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("Sense Exception",e)
        pass                
    return tracks,img

def postprocess_prediction(postprocessors,prediction):
    post_processed_data = {}
    for post_processor in postprocessors:
        post_processed_data.update(post_processor(prediction))
    return {'prediction': prediction, **post_processed_data}

def get_image(image,inference_engine) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Capture image from video stream frame-by-frame.
    The captured image and a scaled copy of the image are returned.
    """
    # ret, img = image
    # if ret:
    img_copy = image.copy()
        # if self.preserve_aspect_ratio:
    img_copy = pad_to_square(image)
    scaled_img = cv2.resize(img_copy, inference_engine.expected_frame_size) if inference_engine.expected_frame_size else image
    return image, scaled_img
    # else:
    #     # Could not grab another frame (file ended?)
    #     return None

def pad_to_square(img):
    """Pad an image to the shape of a square with borders."""
    square_size = max(img.shape[0:2])
    pad_top = int((square_size - img.shape[0]) / 2)
    pad_bottom = square_size - img.shape[0] - pad_top
    pad_left = int((square_size - img.shape[1]) / 2)
    pad_right = square_size - img.shape[1] - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)


def start_inference(inference_engine):
    global clip, frame_index
    print("Starting inference")
    clip = np.random.randn(
        1,
        inference_engine.step_size,
        inference_engine.expected_frame_size[0],
        inference_engine.expected_frame_size[1],
        3
    )
    frame_index = 0
    inference_engine.start()

