import traceback
import cv2
import numpy as np
import sys
from datetime import datetime
import os

# Different OpenCV algorithms
# Status: working

skip_frames = 30*4
previous_grey = None
hsv = None
hsv_roi = None
roi_hist = None
term_criteria = None
x = 200
y = 350
w = 150
h = 150

frameCnt = 0

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

black = (0, 0, 0)

def init_model(transform):
    if transform == 'orb':
        featuresDetector = cv2.ORB_create(nfeatures=1500)
        return featuresDetector, None
    elif transform == 'sift':
        try:
            sift = cv2.xfeatures2d.SIFT_create() 
        except:
            sift = cv2.SIFT_create()
        return sift, None
    elif transform == 'fast':
        fast = cv2.FastFeatureDetector_create()
        return fast, None
    elif transform == 'lkt':
        lk_params = dict( winSize  = (15, 15),#(15, 15),
                          maxLevel = 3,#2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.01))

        feature_params = dict( maxCorners = 5000, #500,
                              qualityLevel = 0.1, #0.3,
                              minDistance = 3, #7,
                              blockSize = 3 ) #7 )
        track_len = 25
        detect_interval = 15
        tracks = []                              
        return (lk_params,feature_params,track_len,detect_interval,tracks), None

    return None, None


def process_image(transform,processing_model,img):
    global previous_grey, hsv, skip_frames,hsv_roi,roi_hist, term_criteria,x, y, w, h,frameCnt
    tracks = []
    frameCnt = frameCnt+1
    try:
        if transform == 'edges':
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
        elif transform == 'cartoon':
          # prepare color
          img_color = cv2.pyrDown(cv2.pyrDown(img))
          for _ in range(6):
              img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
          img_color = cv2.pyrUp(cv2.pyrUp(img_color))

          # prepare edges
          img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          img_edges = cv2.adaptiveThreshold(
              cv2.medianBlur(img_edges, 7), 255,
              cv2.ADAPTIVE_THRESH_MEAN_C,
              cv2.THRESH_BINARY, 9, 2)
          img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

          # combine color and edges
          img = cv2.bitwise_and(img_color, img_edges)

        elif transform == 'detect-color':           
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Every color except white
            low = np.array([0, 42, 0])
            high = np.array([179, 255, 255])
            mask = cv2.inRange(hsv, low, high)
            new_img = cv2.bitwise_and(img, img, mask=mask)
            img = new_img

        elif transform == 'contours':           
            image = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            blurred_frame = image #cv2.GaussianBlur(image, (5, 5), 0)
            gray = cv2.cvtColor(blurred_frame, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            tracks = contours

            img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
            # for contour in contours:
            #   area = cv2.contourArea(contour)
            #   if area > 500:
            #     cv2.drawContours(img, contour, -1, (0, 255, 0), 3)

        elif transform == 'dense-of':
          if previous_grey is None:
            previous_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(img)
            hsv[...,1] = 255
          else:
            img1 = img.copy()
            try:
                next = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                previous_grey,img = drawDenseOpticalFlow(previous_grey,next,hsv)
            except:
                img = img1
                previous_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = np.zeros_like(img)
                hsv[...,1] = 255

        elif transform == 'sift': 
          tracks, img = drawSIFT(img,processing_model)
        elif transform == 'fast': 
          tracks, img = drawFAST(img,processing_model)
        elif transform == 'orb':
          featuresDetector = processing_model
          gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          featuresDetector = cv2.ORB_create(nfeatures=1500)

          keypoints, descriptors = featuresDetector.detectAndCompute(gray, None)

          tracks = keypoints
          img = cv2.drawKeypoints(img, keypoints, None)  

        elif transform == 'mean-shift':
          # perform mean shift tracking
          try:
            if skip_frames>0:
                skip_frames=skip_frames-1
                if(skip_frames==0):
                    roi = img[y: y + h, x: x + w]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                    term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                _, track_window = cv2.meanShift(mask, (x, y, w, h), term_criteria)
                x, y, w, h = track_window
          except Exception as ex:
            print(ex)

          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        elif transform == 'rotate':
            # rotate image
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frameCnt * 5, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

        elif transform == 'lkt':
            (lk_params,feature_params,track_len,detect_interval,tracks) = processing_model
            # frame = img
            frame_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            vis = img

            if len(tracks) > 0:
                # try:
                img0, img1 = previous_grey, frame_gray
                p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    # cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.circle(vis, (int(x), int(y)), 3, (0,0, 255), 2)

                tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
                # draw_str(vis, (20, 20), 'track count: %5d FPS = %0.2f' % (len(tracks), fpsValue))
                # except:
                #     # tracks = []
                #     pass

            if frameCnt % detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        tracks.append([(x, y)])
            previous_grey = frame_gray
            img = vis
        if transform == 'sbs':
            # black = np.zeros((900,1600), dtype = "uint8")
            # h,w = black.shape
            # img = cv2.cvtColor(black,cv2.COLOR_GRAY2RGB)
            img = img 
        elif transform == 'sbs-rg':
            black = np.zeros((900,1600), dtype = "uint8")
            h,w = black.shape
            #extract blue channel
            # blue_channel = img[:,:,0]
            #extract green channel
            green_channel = img[:,:,1]
            #extract red channel
            red_channel = img[:,:,2]

            ih,iw = red_channel.shape
            # print(h,w, ih,iw)
            # temp = np.concatenate((green_channel,red_channel), axis = 1)
            # h,w = black.shape

            diff = 0
            black[h//2-ih//2 : h//2 + ih//2, w//2 - iw - diff : w//2 - diff] = green_channel #red_channel #green_channel
            black[h//2-ih//2 : h//2 + ih//2, w//2 + diff : w//2 + iw + diff] = red_channel
            # black[119:(h-121), 119:w-121] = temp
            img = cv2.cvtColor(black,cv2.COLOR_GRAY2RGB)             

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("OpenCV Exception",e)
        pass

    return tracks,img

def drawSIFT(image,sift):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  (keypoints, descs) = sift.detectAndCompute(gray, None) 
  #Detect key points #
  keypoints = sift.detect(gray, None) 
  #print("Number of keypoints Detected: ", len(keypoints)) 
  # Draw rich key points on input image 
  image = cv2.drawKeypoints(image, keypoints, 0,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return image
  
def drawFAST(image, fast):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  keypoints = fast.detect(gray, None) 
  #print ("Number of keypoints Detected: ", len(keypoints)) 
  # Draw rich keypoints on input image 
  image = cv2.drawKeypoints(image, keypoints,0, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return keypoints, image
  
def drawDenseOpticalFlow(previous_grey,next,hsv):

    # Computes the dense optical flow using the Gunnar Farneback’s algorithm
    flow = cv2.calcOpticalFlowFarneback(previous_grey, next, 
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # use flow to calculate the magnitude (speed) and angle of motion
    # use these values to calculate the color to reflect speed and angle
    magnitude, angle = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = angle * (180 / (np.pi/2))
    hsv[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return next,final
