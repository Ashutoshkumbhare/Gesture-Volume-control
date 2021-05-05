import cv2
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import HandTrackingModule as htm

############################################
wcam, hcam = 640, 480
############################################

cap = cv2.VideoCapture(1)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0

detector = htm.handdetection(detectionconf=0.7, maxHand=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()  # range (-64.0, 0.0, 1.0)

min_val = volRange[0]
max_val = volRange[1]
volBAR = 400
volPER = 0
area = 0
color_vol = (255, 0, 0)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img, draw=True)
    if len(lmlist) != 0:
        # 1. Filter based on size
        # area of bbox
        #       bbox_width        bbox_height
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100  # to get little small number
        # print(area)

        # when hand will be in perticular rance then only volume will be controled
        if 230 < area < 850:
            # print("in range")
            # print(line_len)
            # 2. Find distance between Index and Thumb
            line_len, img, line_info = detector.findDinstance(4, 8, img)

            cx = line_info[4]
            cy = line_info[5]

            # 3. Convert Volume
            # vol range -65 - 0 hand range 5 - 170
            # now we need to convert/normilize hand range into volume range
            volBAR = np.interp(line_len, [5, 170], [400, 150])
            volPER = np.interp(line_len, [5, 170], [0, 100])
            # print(int(line_len), vol)  # vol range -65 - 0

            # 4. Reduce Resolution to make it smoother
            smoothness = 10
            volPER = smoothness * round(volPER / smoothness)

            # 5. Check fingers status
            fingers_status = detector.fingersUP()
            # print(fingers_status)

            # 6. If pinky is down set volume
            if not fingers_status[4]:  # if finger_status == False
                volume.SetMasterVolumeLevelScalar(volPER / 100,
                                                  None)  # it is normilized between 0-1 so to bring in between 0-100 we divided it by 100
                cv2.circle(img, (cx, cy), 10, (255, 255, 0), cv2.FILLED)
                color_vol = (255, 255, 0)

    # creating volume graphics
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBAR)), (85, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f"{int(volPER)} %", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    current_vol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f"Vol Set:{current_vol}", (400, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color_vol, 3)
    color_vol = (255, 0, 0)

    # FPS claculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps:{int(fps)}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
