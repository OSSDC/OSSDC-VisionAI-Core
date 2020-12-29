# OSSDC-VisionAI-Core
OSSDC Vision AI core functionality

This project has support for Race.OSSDC.org WebRTC based platform, to allow for extensive and quick testing, of computer vision and neural nets algorithms, against live (real life or simulated) or streamed (not live) videos (from Youtube or other datasets).

To contribute follow the approach in video_processing files to add your own algorithm and create a PR to integrate it in this project.

Datasets and pretrained models are available in https://github.com/OSSDC/OSSDC-VisionAI-Datasets project.

# Install prerequisites

- pip install opencv-python # required for all video processors
- pip install opencv-contrib-python # required for video_processing_opencv
- pip install aiortc aiohttp python-socketio # required for WebRTC
- pip install dlib # required for face_landmarks
- pip install pytorch torchvision
- pip install tensorflow-gpu
- pip install youtube-dl # required for YouTube streaming sources

# Install VisionAI Android client app

- Download and install the alpha version of the app from here (Soon I'll provide a new link with the VisionAI app):
    - https://race.ossdc.org/apks/race-ossdc-org-remote.apk

# Demos

- Prerequisite steps before running the python processing scripts
    - Run VisionAI Android app and setup the room name and password and start the WebRTC conference
    - Update room info in signaling_race.py (everytime the room name or password is modified in the VisionAI Android app)

- MiDaS mono depth
    - Run VisionAI Android app and setup the room and start the WebRTC conference
    - Install MiDaS - see install steps in video_processing_midas.py
    - run the MiDaS video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t midas
    - Demo Video
        https://www.youtube.com/watch?v=6a6bqJiZuaM

- DLIB face landmarks
    - Install DLIB and face landmarks pretrained model - see instructions steps in video_processing_face_landmarks.py
    - run the DLIB face landmarks video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t face_landmarks

- OpenCV edges detection
    - run the OpenCV edges video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t opencv.edges
