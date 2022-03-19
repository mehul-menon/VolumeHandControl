import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
from comtypes import COMError


cap = cv2.VideoCapture(0)
pTime = 0
detector = htm.HandDetector(detectCon=0.8) #hand detector class from hand detection model
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
        #print(volRange)
while True:
    success, img = cap.read()
    img = detector.findHands(img)# finds hands
    lmlist = detector.findPosition(img,draw=False)# stores list of landmarks
    if len(lmlist)!=0:
        print(lmlist[4],lmlist[8])
        x1, y1 = lmlist[4][1], lmlist[4][2] #co-ordinates of thumb
        x2, y2 = lmlist[8][1], lmlist[8][2] #co-ordinates of index finger
        cx, cy = (x1+x2)//2, (y1+y2)//2 # centre of both index and thumb
        cv2.circle(img, (x1, y1),15, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
        length = math.hypot(x2-x1,y2-y1) # Euclidean distance between index and thumb
        #print(length)
        # hand length range 50-220
        # volume range from -65 to 0
        vol = np.interp(length, [50,220],[minVol,maxVol]) #interpolating distance to find volume
        print(vol)
        if length<50: # to mute, colour of circle changes on muting
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        volume.SetMasterVolumeLevel(vol, None)



    cTime = time.time()
    fps = 1/(cTime-pTime) # finding fps
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
    pTime = cTime
    cv2.imshow('frame',img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
