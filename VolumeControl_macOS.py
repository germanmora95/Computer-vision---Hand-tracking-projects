
import cv2
import mediapipe as mp
import time
import numpy as np
import math
import Module_handtrack as htm
import utils as ut

### PROJECT: COMPUTER VOLUME CONTROL WITH HAND

# We want to do this by tracking our hand, in specific the position of our thumb and index fingers. Depending on the distance of these two fingers, we can increase/decrease volume. 


### Webcam Camera properties
hCam, wCam = 480, 640
min_dist,max_dist = 50,180 # Line distatnces to set min and max volume
min_vol,max_vol = 0,100 # Min and max volume in mac 
cap = ut.read_webcam(hCam=hCam, wCam=wCam)

vol = ut.get_volume()
#ut.get_volume() # This gives us the current volume value in our system (macOS), from 0-100
#ut.set_volume(0) # Set the volume any value between 0-100

t = 0

detector = htm.handDetector(detectionCon=0.7)

while True:

    success, img = cap.read()

    if success:

        img = detector.findHands(img)
        lmList = detector.findPosition(img)  # There will be a total of 21 landmarks with shape [id,cx,cy]. We are interested in landmark 4 and 8 (thumb, index finger)

        if len(lmList) !=0: # When we have deetctions, then display clearly index and thumb landmarks.
            x1,y1 = lmList[4][1],lmList[4][2]
            x2,y2 = lmList[8][1],lmList[8][2]
            len_line = math.hypot(x2-x1,y2-y1) # We calculate the distance between our fingers (of the line)
            cx,cy = (x1+x2)//2, (y1+y2)//2 # Center point of the line
            cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),10)

            vol = np.interp(len_line,[min_dist,max_dist],[min_vol,max_vol]) # We interpolate our distance value between 50-180 to 0-100.
            ut.set_volume(vol)

            

            if len_line < 50:
                cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED) # If the distance between fingers is below 50 (almost touching), then we plot the dot in green. This is to reflect that we activate the button to control volume. 

        else:
            ut.set_volume(min_vol)
            vol = ut.get_volume()

        FPS = 1/(time.time()-t)
        t = time.time()
        
        cv2.rectangle(img, (350,12),(450,32),(0,255,0),3)
        cv2.rectangle(img, (350,12),(350+int(vol),32),(0,255,0),cv2.FILLED)
        cv2.putText(img, "FPS: " + str(int(FPS)), (10,30),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),3 )
        cv2.putText(img, "Volume", (200,30),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),3)
        cv2.putText(img, str(int(vol)), (460,30),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),3)
        cv2.imshow("Frame",img)


    if (cv2.waitKey(1) == 27):  # If we press escape (27) we close the window of the webcam 
        ut.set_volume(15)
        break; 