import mediapipe as mp
import cv2
import math

class handTracker():

    # initialization Function
    def __init__(self, Mode = False, maximumHands = 2, modelComplexity = 1,
                 detConfidence = 0.5, trackConfidence = 0.5):
        self.Mode = Mode
        self.maximumHands = maximumHands
        self.modelComplexity = modelComplexity
        self.detConfidence = detConfidence
        self.trackConfidence = trackConfidence

        # so first thing is that we have to create an object from over class hands (this class is from mediapipe lib)
        self.HandsSol = mp.solutions.hands

        self.hands = self.HandsSol.Hands(self.Mode, self.maximumHands,self.modelComplexity,
                                         self.detConfidence, self.trackConfidence)
        # to draw the line between the landmarks mediapipe also provide solution for that
        self.drawLine = mp.solutions.drawing_utils

    def findAndDrawHands(self, frame):

        RGBimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.outCome = self.hands.process(RGBimage)

        if self.outCome.multi_hand_landmarks:
            for handLandmarks in self.outCome.multi_hand_landmarks:
                self.drawLine.draw_landmarks(frame, handLandmarks,
                                             self.HandsSol.HAND_CONNECTIONS)

        return frame

    def findLandmarks(self, frame, handNo = 0):

        self.landMarksList = []
        x_list = []
        y_list = []
        bbox = []

        if self.outCome.multi_hand_landmarks:
            myHand = self.outCome.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # enumerate returns both id and landmarks
                # print(id)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                x_list.append(cx)
                y_list.append(cy)
                self.landMarksList.append([id, cx, cy])

                ## bbox
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH


        return self.landMarksList, bbox

    def findFingers(self):

        tiIds = [4, 8, 12, 16, 20]
        fingers = []
        # Thumb
        if self.landMarksList[tiIds[0]][1] > self.landMarksList[tiIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.landMarksList[tiIds[id]][2] < self.landMarksList[tiIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.landMarksList[p1][1:]
        x2, y2 = self.landMarksList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
