import traceback
import queue
from pathlib import Path

import cv2
import depthai
import numpy as np

import sys
import argparse
import os
from datetime import datetime, timedelta
from math import cos, sin

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


device = None
cap = None
cam_out = None
detection_in = None
detection_nn = None
reid_in = None
reid_nn = None

bboxes = []
results = {}
results_path = {}
reid_bbox_q = queue.Queue()
next_id = 0

face_bbox_q = queue.Queue()
age_gender_in = None
age_gender_nn = None

#this is needed for gaze estimation visualization
debug=True

useOAKDCam=False

def init_model(transform):
    global device, cap, cam_out, detection_in, detection_nn, reid_in,reid_nn, age_gender_in, age_gender_nn

    if transform == 'pre':
        # sys.path.insert(0, '../depthai-experiments/pedestrian-reidentification')
        device = depthai.Device(create_pipeline_people_reidentification())
        print("Starting pipeline...")
        device.startPipeline()
        cam_out = device.getOutputQueue("cam_out", 1, True)
        detection_in = device.getInputQueue("detection_in")
        detection_nn = device.getOutputQueue("detection_nn")
        reid_in = device.getInputQueue("reid_in")
        reid_nn = device.getOutputQueue("reid_nn")

        # cap = cv2.VideoCapture(str(Path("../depthai-experiments/pedestrian-reidentification/input.mp4").resolve().absolute()))

    elif transform == 'gaze':
        # sys.path.insert(0, '../depthai-experiments/gaze-estimation')
        # import main.Main;
        # cap = cv2.VideoCapture(str(Path("../depthai-experiments/gaze-estimation/demo.mp4").resolve().absolute()))

        model = Main()
        return model, None

    elif transform == 'age-gen':
        device = depthai.Device(create_pipeline_age_gen())
        print("Starting pipeline...")
        device.startPipeline()
        if useOAKDCam:
            cam_out = device.getOutputQueue("cam_out", 1, True)
        else:
            detection_in = device.getInputQueue("detection_in")
        detection_nn = device.getOutputQueue("detection_nn")
        age_gender_in = device.getInputQueue("age_gender_in")
        age_gender_nn = device.getOutputQueue("age_gender_nn")

        # cap = cv2.VideoCapture(str(Path("../depthai-experiments/gen2-age-gender/input.mp4").resolve().absolute()))

    return None, None

def process_image(transform,processing_model,img):
    global useOAKDCam, bboxes, results, results_path, reid_bbox_q, next_id, device, face_bbox_q, age_gender_in, age_gender_nn, cap, cam_out, detection_in, detection_nn, reid_in,reid_nn 
    tracks = []
    try:
        if useOAKDCam:
        #     ret, frame = cap.read()
            frame = np.array(cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)        
        else:
            frame = img

        #pedestrian reidentification https://github.com/luxonis/depthai-experiments/tree/master/pedestrian-reidentification
        if transform == 'pre':

            if frame is not None:
                debug_frame = frame.copy()

                nn_data = depthai.NNData()
                nn_data.setLayer("input", to_planar(frame, (544, 320)))
                detection_in.send(nn_data)
            # else:
            #     return tracks, img

            while detection_nn.has():
                bboxes = np.array(detection_nn.get().getFirstLayerFp16())
                bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
                bboxes = bboxes.reshape((bboxes.size // 7, 7))
                bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

                for raw_bbox in bboxes:
                    bbox = frame_norm_1(frame, raw_bbox)
                    det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    reid_in.send(nn_data)
                    reid_bbox_q.put(bbox)

            while reid_nn.has():
                reid_result = reid_nn.get().getFirstLayerFp16()
                bbox = reid_bbox_q.get()

                for person_id in results:
                    dist = cos_dist(reid_result, results[person_id])
                    if dist > 0.7:
                        result_id = person_id
                        results[person_id] = reid_result
                        break
                else:
                    result_id = next_id
                    results[result_id] = reid_result
                    results_path[result_id] = []
                    next_id += 1

                # if debug:
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                x = (bbox[0] + bbox[2]) // 2
                y = (bbox[1] + bbox[3]) // 2
                results_path[result_id].append([x, y])
                cv2.putText(debug_frame, str(result_id), (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                if len(results_path[result_id]) > 1:
                    cv2.polylines(debug_frame, [np.array(results_path[result_id], dtype=np.int32)], False, (255, 0, 0), 2)
                # else:
                #     print(f"Saw id: {result_id}")
            
            img = debug_frame

        # gaze estimation https://github.com/luxonis/depthai-experiments/tree/master/gaze-estimation
        elif transform == 'gaze':
            model = processing_model            
            model.frame = frame
            tracks, img = model.parse()

        # age gender recognition https://github.com/luxonis/depthai-experiments/tree/master/gen2-age-gender
        elif transform == 'age-gen':
            if frame is not None:
                debug_frame = frame.copy()

                if not useOAKDCam:
                    nn_data = depthai.NNData()
                    nn_data.setLayer("input", to_planar(frame, (300, 300)))
                    detection_in.send(nn_data)

            while detection_nn.has():
                bboxes = np.array(detection_nn.get().getFirstLayerFp16())
                bboxes = bboxes.reshape((bboxes.size // 7, 7))
                bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

                for raw_bbox in bboxes:
                    bbox = frame_norm_1(frame, raw_bbox)
                    det_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    nn_data = depthai.NNData()
                    nn_data.setLayer("data", to_planar(det_frame, (48, 96)))
                    age_gender_in.send(nn_data)
                    face_bbox_q.put(bbox)

            while age_gender_nn.has():
                det = age_gender_nn.get()
                age = int(float(np.squeeze(np.array(det.getLayerFp16('age_conv3')))) * 100)
                gender = np.squeeze(np.array(det.getLayerFp16('prob')))
                gender_str = "female" if gender[0] > gender[1] else "male"
                bbox = face_bbox_q.get()

                while not len(results) < len(bboxes) and len(results) > 0:
                    results.pop(0)
                results.append({
                    "bbox": bbox,
                    "gender": gender_str,
                    "age": age,
                    "ts": time.time()
                })

            results = list(filter(lambda result: time.time() - result["ts"] < 0.2, results))

            if frame is not None:
                for result in results:
                    bbox = result["bbox"]
                    cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                    y = (bbox[1] + bbox[3]) // 2
                    cv2.putText(debug_frame, str(result["age"]), (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))
                    cv2.putText(debug_frame, result["gender"], (bbox[0], y + 20), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255))

            img = debug_frame

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("OAK-D Exception",e)
        pass
                
    return tracks,img


def cos_dist(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def frame_norm_1(frame, bbox):
    return (np.clip(np.array(bbox), 0, 1) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]


def create_pipeline_people_reidentification():
    global useOAKDCam
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    if useOAKDCam:
        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        cam.setPreviewSize(544, 320)
        cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_xout = pipeline.createXLinkOut()
        cam_xout.setStreamName("cam_out")
        cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating Person Detection Neural Network...")
    detection_in = pipeline.createXLinkIn()
    detection_in.setStreamName("detection_in")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("../depthai-experiments/pedestrian-reidentification/models/person-detection-retail-0013.blob").resolve().absolute()))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_in.out.link(detection_nn.input)
    detection_nn.out.link(detection_nn_xout.input)

    # NeuralNetwork
    print("Creating Person Reidentification Neural Network...")
    reid_in = pipeline.createXLinkIn()
    reid_in.setStreamName("reid_in")
    reid_nn = pipeline.createNeuralNetwork()
    reid_nn.setBlobPath(str(Path("../depthai-experiments/pedestrian-reidentification/models/person-reidentification-retail-0031.blob").resolve().absolute()))
    reid_nn_xout = pipeline.createXLinkOut()
    reid_nn_xout.setStreamName("reid_nn")
    reid_in.out.link(reid_nn.input)
    reid_nn.out.link(reid_nn_xout.input)

    print("Pipeline created.")
    return pipeline


def create_pipeline_age_gen():
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

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    detection_nn = pipeline.createNeuralNetwork()
    detection_nn.setBlobPath(str(Path("../depthai-experiments/gen2-age-gender/models/face-detection-retail-0004.blob").resolve().absolute()))
    detection_nn_xout = pipeline.createXLinkOut()
    detection_nn_xout.setStreamName("detection_nn")
    detection_nn.out.link(detection_nn_xout.input)

    if useOAKDCam:
        cam.preview.link(detection_nn.input)
    else:
        detection_in = pipeline.createXLinkIn()
        detection_in.setStreamName("detection_in")
        detection_in.out.link(detection_nn.input)

    # NeuralNetwork
    print("Creating Age Gender Neural Network...")
    age_gender_in = pipeline.createXLinkIn()
    age_gender_in.setStreamName("age_gender_in")
    age_gender_nn = pipeline.createNeuralNetwork()
    age_gender_nn.setBlobPath(str(Path("../depthai-experiments/gen2-age-gender/models/age-gender-recognition-retail-0013.blob").resolve().absolute()))
    age_gender_nn_xout = pipeline.createXLinkOut()
    age_gender_nn_xout.setStreamName("age_gender_nn")
    age_gender_in.out.link(age_gender_nn.input)
    age_gender_nn.out.link(age_gender_nn_xout.input)

    print("Pipeline created.")
    return pipeline


def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }


def to_bbox_result(nn_data):
    arr = to_nn_result(nn_data)
    arr = arr[:np.where(arr == -1)[0][0]]
    arr = arr.reshape((arr.size // 7, 7))
    return arr


def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    has_results = wait_for_results(x_out)
    if not has_results:
        raise RuntimeError("No data from nn!")
    return x_out.get()


def frame_norm(frame, *xy_vals):
    height, width = frame.shape[:2]
    result = []
    for i, val in enumerate(xy_vals):
        if i % 2 == 0:
            result.append(max(0, min(width, int(val * width))))
        else:
            result.append(max(0, min(height, int(val * height))))
    return result


def draw_3d_axis(image, head_pose, origin, size=50):
    roll = head_pose[0] * np.pi / 180
    pitch = head_pose[1] * np.pi / 180
    yaw = -(head_pose[2] * np.pi / 180)

    # X axis (red)
    x1 = size * (cos(yaw) * cos(roll)) + origin[0]
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x1), int(y1)), (0, 0, 255), 3)

    # Y axis (green)
    x2 = size * (-cos(yaw) * sin(roll)) + origin[0]
    y2 = size * (-cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x2), int(y2)), (0, 255, 0), 3)

    # Z axis (blue)
    x3 = size * (-sin(yaw)) + origin[0]
    y3 = size * (cos(yaw) * sin(pitch)) + origin[1]
    cv2.line(image, (origin[0], origin[1]), (int(x3), int(y3)), (255, 0, 0), 2)

    return image


def wait_for_results(queue):
    start = datetime.now()
    while not queue.has():
        if datetime.now() - start > timedelta(seconds=1):
            return False
    return True

class Main:
    def __init__(self, file=None, camera=False):
        print("Loading pipeline...")
        self.file = file
        self.camera = camera
        self.create_pipeline()
        self.start_pipeline()

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = self.pipeline.createColorCamera()
            cam.setPreviewSize(300, 300)
            cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setInterleaved(False)
            cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            cam_xout = self.pipeline.createXLinkOut()
            cam_xout.setStreamName("cam_out")
            cam.preview.link(cam_xout.input)


        # NeuralNetwork
        print("Creating Face Detection Neural Network...")
        face_in = self.pipeline.createXLinkIn()
        face_in.setStreamName("face_in")
        face_nn = self.pipeline.createNeuralNetwork()
        face_nn.setBlobPath(str(Path("../depthai-experiments/gaze-estimation/models/face-detection-retail-0004/face-detection-retail-0004.blob").resolve().absolute()))
        face_nn_xout = self.pipeline.createXLinkOut()
        face_nn_xout.setStreamName("face_nn")
        face_in.out.link(face_nn.input)
        face_nn.out.link(face_nn_xout.input)
        
        # NeuralNetwork
        print("Creating Landmarks Detection Neural Network...")
        land_nn = self.pipeline.createNeuralNetwork()
        land_nn.setBlobPath(
            str(Path("../depthai-experiments/gaze-estimation/models/landmarks-regression-retail-0009/landmarks-regression-retail-0009.blob").resolve().absolute())
        )
        land_nn_xin = self.pipeline.createXLinkIn()
        land_nn_xin.setStreamName("landmark_in")
        land_nn_xin.out.link(land_nn.input)
        land_nn_xout = self.pipeline.createXLinkOut()
        land_nn_xout.setStreamName("landmark_nn")
        land_nn.out.link(land_nn_xout.input)

        # NeuralNetwork
        print("Creating Head Pose Neural Network...")
        pose_nn = self.pipeline.createNeuralNetwork()
        pose_nn.setBlobPath(
            str(Path("../depthai-experiments/gaze-estimation/models/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.blob").resolve().absolute())
        )
        pose_nn_xin = self.pipeline.createXLinkIn()
        pose_nn_xin.setStreamName("pose_in")
        pose_nn_xin.out.link(pose_nn.input)
        pose_nn_xout = self.pipeline.createXLinkOut()
        pose_nn_xout.setStreamName("pose_nn")
        pose_nn.out.link(pose_nn_xout.input)

        # NeuralNetwork
        print("Creating Gaze Estimation Neural Network...")
        gaze_nn = self.pipeline.createNeuralNetwork()
        gaze_nn.setBlobPath(
            str(Path("../depthai-experiments/gaze-estimation/models/gaze-estimation-adas-0002/gaze-estimation-adas-0002.blob").resolve().absolute())
        )
        gaze_nn_xin = self.pipeline.createXLinkIn()
        gaze_nn_xin.setStreamName("gaze_in")
        gaze_nn_xin.out.link(gaze_nn.input)
        gaze_nn_xout = self.pipeline.createXLinkOut()
        gaze_nn_xout.setStreamName("gaze_nn")
        gaze_nn.out.link(gaze_nn_xout.input)

        print("Pipeline created.")

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")
        self.device.startPipeline()
        self.face_in = self.device.getInputQueue("face_in")
        self.face_nn = self.device.getOutputQueue("face_nn")
        self.land_in = self.device.getInputQueue("landmark_in")
        self.land_nn = self.device.getOutputQueue("landmark_nn")
        self.pose_in = self.device.getInputQueue("pose_in")
        self.pose_nn = self.device.getOutputQueue("pose_nn")
        self.gaze_in = self.device.getInputQueue("gaze_in")
        self.gaze_nn = self.device.getOutputQueue("gaze_nn")
        if self.camera:
            self.cam_out = self.device.getOutputQueue("cam_out", 1, True)

    def full_frame_cords(self, cords):
        original_cords = self.face_coords[0]
        return [
            original_cords[0 if i % 2 == 0 else 1] + val
            for i, val in enumerate(cords)
        ]

    def full_frame_bbox(self, bbox):
        relative_cords = self.full_frame_cords(bbox)
        height, width = self.frame.shape[:2]
        y_min = max(0, relative_cords[1])
        y_max = min(height, relative_cords[3])
        x_min = max(0, relative_cords[0])
        x_max = min(width, relative_cords[2])
        result_frame = self.frame[y_min:y_max, x_min:x_max]
        return result_frame, relative_cords

    def draw_bbox(self, bbox, color):
        cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    def run_face(self):
        nn_data = run_nn(self.face_in, self.face_nn, {"data": to_planar(self.frame, (300, 300))})
        results = to_bbox_result(nn_data)
        self.face_coords = [
            frame_norm(self.frame, *obj[3:7])
            for obj in results
            if obj[2] > 0.4
        ]
        if len(self.face_coords) == 0:
            return False
        self.face_frame = self.frame[
            self.face_coords[0][1]:self.face_coords[0][3],
            self.face_coords[0][0]:self.face_coords[0][2]
        ]
        if debug:
            for bbox in self.face_coords:
                self.draw_bbox(bbox, (10, 245, 10))
        return True

    def run_landmark(self):
        nn_data = run_nn(self.land_in, self.land_nn, {"0": to_planar(self.face_frame, (48, 48))})
        out = frame_norm(self.face_frame, *to_nn_result(nn_data))
        raw_left, raw_right, raw_nose = out[:2], out[2:4], out[4:6]

        self.left_eye_image, self.left_eye_bbox = self.full_frame_bbox([
            raw_left[0] - 30, raw_left[1] - 30, raw_left[0] + 30, raw_left[1] + 30
        ])
        self.right_eye_image, self.right_eye_bbox = self.full_frame_bbox([
            raw_right[0] - 30, raw_right[1] - 30, raw_right[0] + 30, raw_right[1] + 30
        ])
        self.nose = self.full_frame_cords(raw_nose)

        if debug:
            cv2.circle(self.debug_frame, (self.nose[0], self.nose[1]), 2, (0, 255, 0), thickness=5, lineType=8, shift=0)
            self.draw_bbox(self.right_eye_bbox, (245, 10, 10))
            self.draw_bbox(self.left_eye_bbox, (245, 10, 10))

    def run_pose(self):
        nn_data = run_nn(self.pose_in, self.pose_nn, {"data": to_planar(self.face_frame, (60, 60))})

        self.pose = [val[0] for val in to_tensor_result(nn_data).values()]

        if debug:
            draw_3d_axis(self.debug_frame, self.pose, self.nose)

    def run_gaze(self):
        nn_data = run_nn(self.gaze_in, self.gaze_nn, {
            "lefy_eye_image": to_planar(self.left_eye_image, (60, 60)),
            "right_eye_image": to_planar(self.right_eye_image, (60, 60)),
            "head_pose_angles": self.pose,
        })

        self.gaze = to_nn_result(nn_data)

        if debug:
            re_x = (self.right_eye_bbox[0] + self.right_eye_bbox[2]) // 2
            re_y = (self.right_eye_bbox[1] + self.right_eye_bbox[3]) // 2
            le_x = (self.left_eye_bbox[0] + self.left_eye_bbox[2]) // 2
            le_y = (self.left_eye_bbox[1] + self.left_eye_bbox[3]) // 2

            x, y = (self.gaze * 100).astype(int)[:2]
            cv2.arrowedLine(self.debug_frame, (le_x, le_y), (le_x + x, le_y - y), (255, 0, 255), 3)
            cv2.arrowedLine(self.debug_frame, (re_x, re_y), (re_x + x, re_y - y), (255, 0, 255), 3)

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        face_success = self.run_face()
        if face_success:
            self.run_landmark()
            self.run_pose()
            self.run_gaze()
            # print(self.gaze)

        # if debug:
        #     aspect_ratio = self.frame.shape[1] / self.frame.shape[0]
        #     cv2.imshow("Camera_view", cv2.resize(self.debug_frame, ( int(900),  int(900 / aspect_ratio))))
        #     if cv2.waitKey(1) == ord('q'):
        #         cv2.destroyAllWindows()
        #         raise StopIteration()
        if debug:
            return self.gaze, self.debug_frame

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

        cap.release()

    def run_camera(self):
        while True:
            self.frame = np.array(self.cam_out.get().getData()).reshape((3, 300, 300)).transpose(1, 2, 0).astype(np.uint8)
            try:
                self.parse()
            except StopIteration:
                break


    def run(self):
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device

