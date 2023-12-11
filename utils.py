import cv2
import mediapipe as mp
import time
import numpy as np
import math
import osascript
import os

def read_webcam(camNum = 0, hCam = 720, wCam = 1280):

    cap = cv2.VideoCapture(camNum)
    cap.set(3,wCam)
    cap.set(4,hCam)

    return cap


def get_volume(): # This gives us the current volume value in our system (macOS), from 0-100
    result = osascript.osascript('get volume settings')
    volInfo = result[1].split(',')
    outputVol = volInfo[0].replace('output volume:', '')
    return outputVol


def set_volume(targetVol= 0):

    vol = "set volume output volume " + str(targetVol)
    osascript.osascript(vol)

def get_images_path(path):

    List_img = (os.listdir(path))
    List_img.sort()

    Header_img = []
    i=0
    for image in List_img: # The idea here is to take all images in a list so we can show how many fingers we have up next to the webcam feed.
        if i>0:
            h_img = cv2.imread(path+image)
            Header_img.append(h_img)
        i+=1
    return Header_img


def mode_selection(total_fingers,finger_up): # If we have two fingers up (index and middle), we got to selection mode, if we have index finger up, we go to drawing mode.

    if total_fingers == 2 and finger_up[1] == 1 and finger_up[2] == 1:
        sel_mode = 1
    elif total_fingers == 1 and finger_up[1] == 1:
        sel_mode = 0
    else:
        sel_mode = -1

    return sel_mode

def idx_selection(px,py, idx, brush_size): # Based on the header picture, if the index finger is inside the rectangle, we change the header image and colour that corresponding rectanglep's colour.



    if px >= 200 and px <= 350 and py < 136:
        idx = 0
        brush_size = 10
    elif px >= 500 and px <= 650 and py < 136:
        idx = 1
        brush_size = 10
    elif px >= 800 and px <= 950 and py < 136:
        idx = 2
        brush_size = 10
    elif px >= 1100 and px <= 1250 and py < 136:
        idx = 3
        brush_size = 50

    return idx, brush_size

def overlay_painting(img,imgCanvas): # We want to overlay rgb webcam image with the drawing canvas.

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY) # We convert the canvas in grayscale.
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV) # We binarise the image and invert it -> drawing is black and rest is white. 
    imgInv =  cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR) # We convert the canvas back into BGR to overlay (same dimensions) with the webcam image.
    img = cv2.bitwise_or(cv2.bitwise_and(img,imgInv), imgCanvas) # The bitwise and draws black in the webcam image wherever we draw, and the or operation sums the canvas (with the colour) to the webcam image.

    return img