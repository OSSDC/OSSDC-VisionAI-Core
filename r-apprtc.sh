#!/bin/bash
source ~/.virtualenvs/gen2/bin/activate
python3 apprtc_webrtc_processing.py -t  $1 --room $2  --skipFrames $3
