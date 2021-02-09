# OSSDC VisionAI

**A set of computer vision and artificial intelligence algorithms for robotics and self driving cars**

This project has support for Race.OSSDC.org WebRTC based platform, to allow for extensive and quick testing, of computer vision and neural nets algorithms, against live (real life or simulated) or streamed (not live) videos (from Youtube or other datasets).

To contribute follow the approach in video_processing files to add your own algorithm and create a PR to integrate it in this project.

Datasets and pretrained models are available in https://github.com/OSSDC/OSSDC-VisionAI-Datasets project.

## Install prerequisites

- pip install opencv-python # required for all video processors
- pip install opencv-contrib-python # required for video_processing_opencv
- pip install aiortc aiohttp python-socketio # required for WebRTC
- pip install dlib # required for face_landmarks
- pip install pytorch torchvision
- pip install tensorflow-gpu
- pip install youtube-dl # required for YouTube streaming sources

## Install OSSDC VisionAI Android client app

- Download and install the alpha version from here:
    - https://github.com/OSSDC/OSSDC-VisionAI-Mobile/releases/

## Demos

- Prerequisite steps every time before running the python video processing scripts
    - Run VisionAI Android app and setup the room name and password and start the WebRTC conference
    - Update room info in signaling_race.py (everytime the room name or password is modified in the VisionAI Android app)

###### OAK-D Spacial AI camera 

<p align="center">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/oak-d_gaze_1.gif" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/oak-d_pedestrian_reidentification_1.gif" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/oak-d_ssd_1.gif" width="300px">
</p>

(**Gaze estimation** video can be found [here](https://www.youtube.com/watch?v=xMgNWRWytOk))

(**Pedestrian re-identification** video can be found [here](https://www.youtube.com/watch?v=pB0BpHieu3Y))

(**SSD object detection** video can be found [here]())

- OAK-D gaze estimation demo, the proceessing is done on Luxonis OAK-D camera vision processing unit https://store.opencv.ai/products/oak-d
    - Install OAK-D DepthAI - see install steps in video_processing_oakd.py
    - run the OAK-D video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t oakd.gaze
    - Demo video
    
        Gaze estimation demo with processing done on Luxonis OAK-D camera processor (processing at 10 FPS on 486 x 1062 video, streamed at 30 FPS) 
        
        https://www.youtube.com/watch?v=xMgNWRWytOk

- OAK-D people reidentification demo, the proceessing is done on Luxonis OAK-D camera vision processing unit https://store.opencv.ai/products/oak-d
    - Run VisionAI Android app and setup the room and start the WebRTC conference
    - Install OAK-D DepthAI - see install steps in video_processing_oakd.py
    - run the OAK-D video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t oakd.pre
    - Demo video
    
        People reidentification demo with processing done on Luxonis OAK-D camera processor (processing at 9 FPS on 486 x 1062 video, streamed at 30 FPS) 
        
        https://www.youtube.com/watch?v=pB0BpHieu3Y

- OAK-D age and genrer recognition demo, the proceessing is done on Luxonis OAK-D camera vision processing unit https://store.opencv.ai/products/oak-d
    - Install OAK-D DepthAI - see install steps in video_processing_oakd.py
    - run the OAK-D video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t oakd.age-gen
    - Demo video
    
        Upcomming    

###### MiDaS mono depth

<p align="center">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/midas_person_1.gif" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/raw/master/docs/midas_night_1.gif" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/midas_objects_1.gif" width="300px">
</p>

(**MiDaS mono-depth person demo** video can be found [here](https://www.youtube.com/watch?v=xMgNWRWytOk))

(**MiDaS mono-depth night walk demo** video can be found [here](https://www.youtube.com/watch?v=T0ZnW1crm7M))

(**MiDaS mono-depth objects demo** video can be found [here]())

- MiDaS mono depth, processing is done on Nvidia GPU
    - Run VisionAI Android app and setup the room and start the WebRTC conference
    - Install MiDaS - see install steps in video_processing_midas.py
    - run the MiDaS video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t midas
    - Demo Videos

        Mono depth over WebRTC using Race.OSSDC.org platform
        
        https://www.youtube.com/watch?v=6a6bqJiZuaM

        OSSDC VisionAI MiDaS Mono Depth - night demo
        
        https://www.youtube.com/watch?v=T0ZnW1crm7M
        
- DLIB face landmarks, processing is done on CPU
    - Install DLIB and face landmarks pretrained model - see instructions steps in video_processing_face_landmarks.py
    - run the DLIB face landmarks video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t face_landmarks

- OpenCV edges detection, processing is done on CPU
    - run the OpenCV edges video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t opencv.edges
