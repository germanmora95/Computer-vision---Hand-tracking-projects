import cv2
import mediapipe as mp
import time


class handDetector():
     
    def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionCon = 0.5, trackCon = 0.5):
         
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands # The Hands class in MediaPipe is designed for hand tracking tasks, allowing you to detect and track hands in images or video streams.
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.complexity, self.detectionCon,self.trackCon) #  This line creates an instance of the Hands class and assigns it to the variable hands.  By default it tracks (instead of just detect) and maximum 2 hands, and has detection threshold of 0.5
        self.mpDraw = mp.solutions.drawing_utils # Code to draw the 21 points in the hand

    def findHands(self,img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Need to convert to RGB since we always import in BGR
        self.results = self.hands.process(imgRGB) # This process the data

        if self.results.multi_hand_landmarks: # This checks if there's multiple landmarks detected in the hand.
                for handLms in self.results.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)  # For every landmark handLms, you draw them. The third command is optional and shows connections between hands.  
     
        return img

    def findPosition(self,img,handID=0,draw=True): # We retrieve the position of a given landmark and give the option to draw it.
        
        h,w,c = img.shape   
        self.lmList = [] # List of location of all landmarks.

        if self.results.multi_hand_landmarks: # We check again if we have landmarks.

            myHand = self.results.multi_hand_landmarks[handID] # We just take the landmarks of the selected hand

            for id, lm in enumerate(myHand.landmark): # Each ID has a lanmark, and that has a x,y,z coordinates. The coordinates are given in decimal (the ratio of the image), but we need pixels.
                        
                        cx,cy = int(lm.x*w), int(lm.y*h) # We get the centre of points of each landmark, which then we can store in a list.  
                        self.lmList.append([id,cx,cy])  # We append the ID, cx, cy in a list of lists.

                        if draw:
                            cv2.circle(img,(cx,cy),7,(255,0,255),cv2.FILLED)

                    
        return self.lmList
    
    def finger_counter(self, fingertips):
        
        finger_up = []
        for ID in range(0,len(fingertips)):
            if ID == 0 and self.lmList[fingertips[ID]][1] > self.lmList[fingertips[ID]+16][1]: # Compare the pinky with the thumb so the behaviour is symmetric for each hand. 
                if self.lmList[fingertips[ID]][1] > self.lmList[fingertips[ID]-1][1]: # If the thumb tip is to the right of its inferior landmark , then it is up.
                    finger_up.append(1)
                else:
                    finger_up.append(0)
            elif ID == 0 and self.lmList[fingertips[ID]][1] < self.lmList[fingertips[ID]+16][1]: # Compare the pinky with the thumb for the other hand. 
                if self.lmList[fingertips[ID]][1] < self.lmList[fingertips[ID]-1][1]: # If the thumb tip is to the left of its inferior landmark then it is up.
                    finger_up.append(1)
                else:
                    finger_up.append(0)
            else:
                if self.lmList[fingertips[ID]][2] < self.lmList[fingertips[ID]-2][2]: # if finger tip is above 2 landmarks below in the same finger, it is open finger.
                    finger_up.append(1)
                else:
                    finger_up.append(0)

        total_fingers = finger_up.count(1)

        return total_fingers, finger_up


def main():

    cap = cv2.VideoCapture(0)
    t = 0

    detector = handDetector()

    while True:
        success, img = cap.read()
        
        img = detector.findHands(img)
        lmList = detector.findPosition(img,draw=False) #

        if len(lmList) !=0:
            print(lmList[4]) # Display thumb point value. 

        FPS = 1/(time.time()-t)
        t = time.time()
        

        cv2.putText(img, "FPS: " + str(int(FPS)), (10,30),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,255),3 )
        cv2.imshow("Frame",img)


        if (cv2.waitKey(1) == 27):  # If we press escape (27) we close the window of the webcam 
            break; 


if __name__ == "__main__":
    main()