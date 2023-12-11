
import cv2
import numpy as np
import mediapipe as mp
import time
import Module_handtrack as htm
import os
import utils as ut

detector = htm.handDetector( detectionCon = 0.8, trackCon = 0.8)
cap = cv2.VideoCapture(0)
t = 0

path_resources = "/Users/german/Library/CloudStorage/Dropbox/Projects/CV_Advanced_Py_Cpp/Py/01_Handtracking/Virtualpaint_resources/Images/"
Colours = [(36,28,237),(63,198,141),(225,170,39),(0,0,0)]
Header_imgs = ut.get_images_path(path_resources)

mode = -1 # By defaul, we do not do anythting (-1). Draw = 0, selection = 1
idx = 0 # By default, the selection mode starts with red. 
brush_size = 10
fingertips = [4,8,12,16,20] #IDs of finger tips (thumb to pinky)
h,w,c = np.shape(Header_imgs[idx])

success, img = cap.read()
hcam,wcam,ccam = np.shape(img)
imgCanvas = np.zeros((hcam,wcam,ccam), np.uint8)

while True:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img, draw = False)
    lmList = detector.findPosition(img, draw = False)

    if len(lmList) !=0:


        total_fingers, finger_up = detector.finger_counter(fingertips)

        mode = ut.mode_selection(total_fingers,finger_up)

        if mode == 1: # If selection mode is activated, we draw a square around the fingers. 

            px0,py0 = 0,0 # Initialise previous points for line drawing.

            cv2.rectangle(img, (lmList[fingertips[1]][1],lmList[fingertips[1]][2]),(lmList[fingertips[2]][1],lmList[fingertips[2]][2]),Colours[idx],cv2.FILLED)
            px, py = lmList[fingertips[1]][1],lmList[fingertips[1]][2]
            
            idx, brush_size = ut.idx_selection(px,py,idx,brush_size)

            
        
        if mode == 0:


            px, py = lmList[fingertips[1]][1],lmList[fingertips[1]][2]

            if px0 == 0 and py0 == 0:
                px0,py0 = px,py
           
            cv2.line(img, (px0,py0),(px,py), Colours[idx], brush_size,  cv2.FILLED)
            cv2.line(imgCanvas, (px0,py0),(px,py), Colours[idx], brush_size, cv2.FILLED)

            px0,py0 = px,py


    img = ut.overlay_painting(img,imgCanvas)

        
        




    
    img[0:h,0:w] = Header_imgs[idx]

        

    FPS = 1/(time.time()-t)
    t = time.time()
    

    cv2.putText(img, "FPS: " + str(int(FPS)), (10,700),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),3 )
    cv2.imshow("Frame",img)


    if (cv2.waitKey(1) == 27):  # If we press escape (27) we close the window of the webcam 
        break; 