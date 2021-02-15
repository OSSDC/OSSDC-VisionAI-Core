#OpenCV Reel
#working
# declare -a algos=("opencv.edges" "opencv.fast" "opencv.orb" "opencv.rotate" "opencv.detect-color" "opencv.mean-shift" "opencv.dense-of")
#not working
# declare -a algos=( "opencv.cartoon" "opencv.sift")

#OAK-D reel
# declare -a algos=("oakd.ssd" "oakd.gaze" "oakd.age-gen" 'oakd.pre' 'oakd_palm')

#SDC
# declare -a algos=('oakd.ssd' 'yolact' "midas" "midas21" 'opencv.dense-of' "opencv.edges")

# declare -a algos=('opencv.edges')
#declare -a algos=('head_pose')
# declare -a algos=('posenet')
#declare -a algos=('face_landmarks')
# declare -a algos=('oakd.gaze')
# declare -a algos=('oakd.ssd')
# declare -a algos=('opencv.edges+oakd.ssd')
# declare -a algos=('oakd.gaze+oakd.ssd') #not working
# declare -a algos=('oakd.age-gen+oakd.ssd') #not working
#declare -a algos=('midas')
# declare -a algos=('midas21')
#declare -a algos=('midas+midas21')
# declare -a algos=('oakd.ssd+yolo5')
# declare -a algos=('opencv.edges+oakd.ssd+midas')
# declare -a algos=('opencv.edges+midas+yolact')
# declare -a algos=('opencv.edges+midas21')
# declare -a algos=('oakd.pre')
# declare -a algos=('yolo5')
# declare -a algos=('opencv.edges+yolo5')
# declare -a algos=('opencv.edges+yolact+yolo5') #not working
# declare -a algos=('opencv.edges+midas+yolo5')
# declare -a algos=('opencv.edges+midas21+yolo5')
# declare -a algos=('midas+yolo5')
# declare -a algos=('midas' 'yolo5' 'midas+yolo5')
# declare -a algos=('oakd.gaze+opencv.edges')
# declare -a algos=('oakd.gaze+opencv.edges+midas')
# declare -a algos=('midas+opencv.edges+oakd.gaze')
# declare -a algos=('midas21+opencv.edges')
# declare -a algos=('opencv.edges+yolact')
# declare -a algos=('yolact+opencv.edges')
# declare -a algos=('yolact+midas')
#declare -a algos=('yolact')
# declare -a algos=('oakd_palm')
# declare -a algos=('opencv.edges')

# declare -a algos=('sense.gesture' 'sense.fitness')
# declare -a algos=('sense.gesture')
# declare -a algos=('sense.fitness')

# declare -a algos=('gaze_est')

# declare -a algos=('mediapipe.facemesh')
# declare -a algos=('mediapipe.hands')
# declare -a algos=('mediapipe.pose')
# declare -a algos=('mediapipe.holistic')

declare -a algos=('mediapipe.holistic' 'mediapipe.facemesh' 'mediapipe.hands' 'mediapipe.pose')

pkill -9 -f race
sleep 1
pkill -9 -f race
sleep 2

echo $1
echo $2
echo $3

for i in {1..5..1}
do
   # Iterate the string array using for loop
   for val in ${algos[@]}; do
      echo $val
      python3 race-ossdc-org_webrtc_processing.py -t $val --room $1 --videoUrl "$2" --skipFramesCnt $3 --twitchStreamKey "$4" -mt small&
      # python3.8 race-ossdc-org_no_webrtc_processing.py -t $val --room $1 --videoUrl "$2" --skipFramesCnt $3 --twitchStreamKey "$4" &
      #-rec "$(date +"../%Y_%m_%d_%I_%M_%p").mp4"&
      sleep 60
      pkill -9 -f race
      sleep 1
      pkill -9 -f race
      sleep 2
   done
done
