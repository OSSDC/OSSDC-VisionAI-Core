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
          traks, img = drawSIFT(img,processing_model)
        elif transform == 'fast': 
          traks, img = drawFAST(img,processing_model)
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
