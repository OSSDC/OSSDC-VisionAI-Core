#!/bin/bash
while :; do
#   /c/apps/ffmpeg/bin/ffplay -fs -autoexit -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
   #ffplay -fs -autoexit -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen
   ffplay -autoexit -fflags nobuffer -f mjpeg tcp://0.0.0.0:45654?listen

done

