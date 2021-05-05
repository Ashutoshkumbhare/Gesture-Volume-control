import cv2
import mediapipe as mp
import time
import math


class handdetection():
    def __init__(self, mode=False, maxHand=2, detectionconf=0.5, trackconf=0.5):
        self.mode = mode
        self.maxHand = maxHand
        self.detectionconf = detectionconf
        self.trackconf = trackconf

        self.mphands = mp.solutions.hands  # creating object
        self.hands = self.mphands.Hands(self.mode, self.maxHand, self.detectionconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)  # will process the RGB image through media-pipeline and return.
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_lms, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, num_of_hand=0, draw=True):

        Xlist = []
        Ylist = []
        bbox = []
        self.li = []
        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[num_of_hand]
            for i, xyz in enumerate(myhands.landmark):
                # print(i, xyz)
                # it will give ous landmarks position in decimal num, we need pickles so converting it into pickles

                # getting actual size of img
                h, w, c = img.shape  # hight, width, center of the image

                # getting exact position wrt pickles
                cx, cy = int(xyz.x * w), int(xyz.y * h)  # (xyz.x)xyz ka x
                # print(i, cx, cy)
                Xlist.append(cx)
                Ylist.append(cy)
                self.li.append([i, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # This will give ous xmin, xmax, ymin, ymax for boundry box
            Xmin, Xmax = min(Xlist), max(Xlist)
            Ymin, Ymax = min(Ylist), max(Ylist)
            bbox = Xmin, Ymin, Xmax, Ymax

            # drawing bbox
            if draw:
                cv2.rectangle(img, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (255, 0, 0), 2)

        return self.li, bbox

    def fingersUP(self):

        fingers_status = []

        # Thumb
        if self.li[self.tipIds[0]][1] > self.li[self.tipIds[0] - 1][1]:
            fingers_status.append(1)
        else:
            fingers_status.append(0)

        # 4 fingers
        for i in range(1, 5):
            if self.li[self.tipIds[i]][2] < self.li[self.tipIds[i] - 1][2]:
                fingers_status.append(1)
            else:
                fingers_status.append(0)

        return fingers_status

    def findDinstance(self, p1, p2, img, draw=True):
        # work: This will find the distance between two fingers
        # return: int value of distance

        x1, y1 = self.li[p1][1], self.li[p1][2]
        x2, y2 = self.li[p2][1], self.li[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 6)
            cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        line_len = math.hypot((x2 - x1), (y2 - y1))
        return line_len, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    detector = handdetection()
    cap = cv2.VideoCapture(1)
    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        landmark_list = detector.findPosition(img)

        if len(landmark_list) != 0:
            print(landmark_list[12])

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
