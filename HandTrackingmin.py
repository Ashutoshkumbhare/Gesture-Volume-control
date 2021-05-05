import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mphanda = mp.solutions.hands  # creating object
hands = mphanda.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)  # will process the RGB image through media-pipeline and return.
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # print(hand_lms.landmark)
            for i, xyz in enumerate(hand_lms.landmark):
                # print(i, xyz)
                # getting actual size of img
                h, w, c = img.shape  # hight, width, center of the image
                cx, cy = int(xyz.x * w), int(xyz.y * h)
                print(i,cx,cy)
                if i==0:
                    cv2.circle(img,(cx,cy),20,(0,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img, hand_lms, mphanda.HAND_CONNECTIONS)

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("image", img)
    cv2.waitKey(1)
