import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import HandTrackingModule as htm
import math

############################################
wcam, hcam = 640, 480
############################################

cap = cv2.VideoCapture(1)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handdetection(detectionconf=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # range (-64.0, 0.0, 1.0)
# volume.SetMasterVolume(-20.8,None)

min_val = volRange[0]
max_val = volRange[1]
volBAR = 400
volPER = 0
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 6)
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        line_len = math.hypot((x2 - x1), (y2 - y1))
        # print(line_len)

        if line_len < 30:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        # vol range -65 - 0 hand range 5 - 170
        # now we need to convert/normilize hand range into volume range
        vol = np.interp(line_len, [5, 170], [min_val, max_val])
        volBAR = np.interp(line_len, [5, 170], [400, 150])
        volPER = np.interp(line_len, [5, 170], [0, 100])
        #print(int(line_len), vol)  # vol range -65 - 0
        volume.SetMasterVolumeLevel(vol, None)

    # creating volume graphics
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBAR)), (85, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPER)} %", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # FPS claculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
