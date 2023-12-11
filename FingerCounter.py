
import cv2
import mediapipe as mp
import time
import numpy as np
import math
import Module_handtrack as htm
import utils as ut
import os

### PROJECT: FINGER COUNTER

### Webcam Camera properties
hCam, wCam = 480, 640

path_resources = "/Users/german/Library/CloudStorage/Dropbox/Projects/CV_Advanced_Py_Cpp/Py/01_Handtracking/Resources/"
List_img = (os.listdir(path_resources))
List_img.sort()

Finger_img = []
for image in List_img: # The idea here is to take all images in a list so we can show how many fingers we have up next to the webcam feed.
    f_img = cv2.imread(path_resources+image)
    h,w,c = f_img.shape
    if h > 200 and h > 200:
        f_img = cv2.resize(f_img, (200,200))
    Finger_img.append(f_img)

h,w,c = Finger_img[0].shape

cap = ut.read_webcam(hCam=hCam, wCam=wCam)

t = 0

detector = htm.handDetector(detectionCon=0.7)
fingertips = [4,8,12,16,20] #IDs of finger tips (thumb to pinky)

while True:

    success, img = cap.read()

    if success:

        img = detector.findHands(img)
        lmList = detector.findPosition(img)  # There will be a total of 21 landmarks with shape [id,cx,cy]. We are interested in landmark 4 and 8 (thumb, index finger)

        
        if len(lmList) !=0: # When we have deetctions, then display clearly index and thumb landmarks.
            finger_up = []
            for ID in range(0,len(fingertips)):
                if ID == 0 and lmList[fingertips[ID]][1] > lmList[fingertips[ID]+16][1]: # Compare the pinky with the thumb so the behaviour is symmetric for each hand. 
                    if lmList[fingertips[ID]][1] > lmList[fingertips[ID]-1][1]: # If the thumb tip is to the right of its inferior landmark , then it is up.
                        finger_up.append(1)
                    else:
                        finger_up.append(0)
                elif ID == 0 and lmList[fingertips[ID]][1] < lmList[fingertips[ID]+16][1]: # Compare the pinky with the thumb for the other hand. 
                    if lmList[fingertips[ID]][1] < lmList[fingertips[ID]-1][1]: # If the thumb tip is to the left of its inferior landmark then it is up.
                        finger_up.append(1)
                    else:
                        finger_up.append(0)
                else:
                    if lmList[fingertips[ID]][2] < lmList[fingertips[ID]-2][2]: # if finger tip is above 2 landmarks below in the same finger, it is open finger.
                        finger_up.append(1)
                    else:
                        finger_up.append(0)

            total_fingers = finger_up.count(1) # We count how many fingers we have up/open and assign the corresponding default image. 
            img[0:h,0:w] = Finger_img[total_fingers-1]


        
        FPS = 1/(time.time()-t)
        t = time.time()

        cv2.putText(img, "FPS: " + str(int(FPS)), (10,30),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),3 )

        cv2.imshow("Frame",img)


    if (cv2.waitKey(1) == 27):  # If we press escape (27) we close the window of the webcam 

        break; 