# OSSDC VisionAI

**A set of computer vision and artificial intelligence algorithms for robotics and self driving cars**

This project has support for Race.OSSDC.org WebRTC based platform, to allow for extensive and quick testing, of computer vision and neural nets algorithms, against live (real life or simulated) or streamed (not live) videos (from Youtube or other datasets).

To contribute follow the approach in video_processing files to add your own algorithm and create a PR to integrate it in this project.

<p align="center">
OSSDC VisionAI Demo Reel - run the algoritms in Google Colab
</p>
<p align="center">
<a href="https://colab.research.google.com/github/OSSDC/OSSDC-VisionAI-Core/blob/master/OSSDC_VisionAI_demo_reel.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open Demo Reel In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

###### OAK-D Spacial AI camera - Demos

<p align="center">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/blob/master/docs/oak-d_gaze_1.gif?raw=true" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/blob/master/docs/oak-d_pedestrian_reidentification_1.gif?raw=true" width="300px">
    <!-- img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/master/docs/gifs/oak-d_ssd_1.gif?raw=true" width="300px" -->
</p>

(**Gaze estimation** video can be found [here](https://www.youtube.com/watch?v=xMgNWRWytOk))

(**Pedestrian re-identification** video can be found [here](https://www.youtube.com/watch?v=pB0BpHieu3Y))

(**SSD object detection** video can be found [here]())

###### MiDaS mono depth - Demos

<p align="center">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/blob/master/docs/midas_person_1.gif?raw=true" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/blob/master/docs/midas_night_1.gif?raw=true" width="300px">
    <img src="https://github.com/OSSDC/OSSDC-VisionAI-Core/blob/master/docs/midas_objects_1.gif?raw=true" width="300px">
</p>

(**MiDaS mono-depth person demo** video can be found [here](https://www.youtube.com/watch?v=6a6bqJiZuaM))

(**MiDaS mono-depth night walk demo** video can be found [here](https://www.youtube.com/watch?v=T0ZnW1crm7M))

(**MiDaS mono-depth objects demo** video can be found [here]())

Datasets and pretrained models are available in https://github.com/OSSDC/OSSDC-VisionAI-Datasets project.

## Install prerequisites

- pip install opencv-python # required for all video processors
- pip install opencv-contrib-python # required for video_processing_opencv
- pip install aiortc aiohttp websockets python-engineio==3.14.2 python-socketio[client]==4.6.0 # required for WebRTC
- pip install dlib # required for face_landmarks
- pip install torch torchvision
- pip install tensorflow-gpu
- pip install youtube-dl # required for YouTube streaming sources

## Install OSSDC VisionAI Android client app

- Download and install the alpha version from here:
    - https://github.com/OSSDC/OSSDC-VisionAI-Mobile/releases/

## Demos

- Prerequisite steps every time before running the python video processing scripts
    - Run VisionAI Android app and setup the room name and password and start the WebRTC conference
    - Update room info in signaling_race.py (everytime the room name or password is modified in the VisionAI Android app)

- DeepMind NFNets demo
    - Install DeepMind NFNets - see install steps in video_processing_deepmind.py or OSSDC_VisionAI_demo_reel.ipynb notebook
    - run the DeepMind NFNets video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t deepmind.nfnets --room {your_room_name}
        - demo-reel.sh {your_room_name} (enable deepmind.nfnets line)    
    - Demo samples images
        https://www.linkedin.com/feed/update/urn:li:activity:6766007580679557120?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A6766007580679557120%2C6768387554418016256%29

- MediaPipe Holistic demo
    - Install MediaPipe - see install steps in video_processing_mediapipe.py or OSSDC_VisionAI_demo_reel.ipynb notebook
    - run the MediaPipe holistic video processor on the video stream from VisionAI Android app
        - python race-ossdc-org_webrtc_processing.py -t mediapipe.holistic --room {your_room_name}
        - demo-reel.sh {your_room_name} (enable mediapipe.holistic line) 
    - Demo video
    
        MediaPipe holistic demo 
        
        Isn't this fun?! MediaPipe Holistic neural net model processed in real time on Google Cloud
        https://www.youtube.com/watch?v=0l9Bb5IC86E

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
