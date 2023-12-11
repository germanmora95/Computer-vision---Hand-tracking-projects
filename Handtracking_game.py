
import cv2
import mediapipe as mp
import time
import Module_handtrack as htm

cap = cv2.VideoCapture(0)
t = 0

detector = htm.handDetector()

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) !=0:
        print(lmList[4])

    FPS = 1/(time.time()-t)
    t = time.time()
    

    cv2.putText(img, "FPS: " + str(int(FPS)), (10,30),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),3 )
    cv2.imshow("Frame",img)


    if (cv2.waitKey(1) == 27):  # If we press escape (27) we close the window of the webcam 
        break; 