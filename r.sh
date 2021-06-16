#!/bin/bash
source ~/.virtualenvs/gen2/bin/activate
python3 race-ossdc-org_webrtc_processing.py -t  $1 --room $2  --skipFrames $3
