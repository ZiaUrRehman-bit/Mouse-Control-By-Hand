import cv2
import numpy as np
import Track_Hand as ht
import time
import pyautogui

#########################################
wCam, hCam = 640, 480
frameR = 100    # Frame Reduction
#########################################

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, wCam)
cam.set(4, hCam)

hand = ht.handTracker(maximumHands=1,
                      detConfidence=0.8,
                      trackConfidence=0.8)

wScr, hScr = pyautogui.size()
# print(wScr, hScr)
# for frame rate


######################## idea ##########################
# So the idea is if the index finger up then the cursor#
# is in moving state and if the middle finger also up  #
# then the mouse is in clicking mode.                  #
# and if the distance between the index and middle fin-#
# -ger is less then certain valuse then click occur.   #
#######################  End ###########################






while True:

    # 1. Find Hand Landmarks

    Success, frame = cam.read()
    frame = hand.findAndDrawHands(frame)

    # 2. Get the tip of middle and index finger
    lm, bbox = hand.findLandmarks(frame)
    if lm:
        x1, y1 = lm[8][1:]
        x2, y2 = lm[12][1:]
        # print(x1,y1, x2, y2)

        # 3. Check which finger is up
        fingers = hand.findFingers()
            # print(fingers)

        # drawing a rectangle to specify region to move mouse cursor
        cv2.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (0, 255, 255), 2)

        # 4. only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert the coordinates

            # x3 = np.interp(x1, (0, wCam), (0, wScr))
            # y3 = np.interp(y1, (0, hCam), (0, hScr))
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen the values
            # 7. move mouse
            # pyautogui.moveTo(x3, y3)
            # we need to flip as if we move right it goes left
            pyautogui.moveTo(wScr-x3, y3)
            cv2.circle(frame, (x1, y1), 15, (0,255, 255), cv2.FILLED)
        # 8. both middle and index finger is up: clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, frame, lineInfo = hand.findDistance(8, 12, frame)
            print(length)
            # 10. click the mouse if distance is short
            if length < 30:
                cv2.circle(frame, (lineInfo[4], lineInfo[5]), 15,
                           (0, 255, 0), cv2.FILLED)
                pyautogui.click(button="left")



    # 11. Display
    cv2.imshow("Mouse Control", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()


