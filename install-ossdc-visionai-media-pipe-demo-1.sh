git clone https://github.com/OSSDC/OSSDC-VisionAI-Core
pip install aiortc aiohttp websockets python-engineio==3.14.2 python-socketio[client]==4.6.0 youtube-dl
pip install mediapipe
cd OSSDC-VisionAI-Core
#./demo-reel.sh YourRoomName

# fix for Yolact with CUDA 11.1
export CUDA_PATH=/usr/local/cuda-11.1/
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PA
