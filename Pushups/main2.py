import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math
import time
from Model import failureRep

counter = 0
direction = 0
rep_start_time = None  # Initialize the variable to track the start time of each rep
repSpeedArray=[]
cap = cv2.VideoCapture('vid2.mp4')
pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

def angles(lmlist, p1, p2, p3, p4, p5, p6, drawpoints):
    global counter
    global direction
    global rep_start_time  

    if len(lmlist) != 0:
        point1 = lmlist[p1]
        point2 = lmlist[p2]
        point3 = lmlist[p3]
        point4 = lmlist[p4]
        point5 = lmlist[p5]
        point6 = lmlist[p6]

        x1, y1 = point1[0:-1]
        x2, y2 = point2[0:-1]
        x3, y3 = point3[0:-1]
        x4, y4 = point4[0:-1]
        x5, y5 = point5[0:-1]
        x6, y6 = point6[0:-1]

        if drawpoints == True:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), 5)
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), 5)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), 5)
            cv2.circle(img, (x2, y2), 15, (0, 255, 0), 5)
            cv2.circle(img, (x3, y3), 10, (255, 0, 255), 5)
            cv2.circle(img, (x3, y3), 15, (0, 255, 0), 5)
            cv2.circle(img, (x4, y4), 10, (255, 0, 255), 5)
            cv2.circle(img, (x4, y4), 15, (0, 255, 0), 5)
            cv2.circle(img, (x5, y5), 10, (255, 0, 255), 5)
            cv2.circle(img, (x5, y5), 15, (0, 255, 0), 5)
            cv2.circle(img, (x6, y6), 10, (255, 0, 255), 5)
            cv2.circle(img, (x6, y6), 15, (0, 255, 0), 5)

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 6)
            cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 6)
            cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 6)
            cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 6)
            cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 6)

        lefthandangle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                                     math.atan2(y1 - y2, x1 - x2))

        righthandangle = math.degrees(math.atan2(y6 - y5, x6 - x5) -
                                      math.atan2(y4 - y5, x4 - x5))

        leftHandAngle = int(np.interp(lefthandangle, [-30, 180], [100, 0]))
        rightHandAngle = int(np.interp(righthandangle, [34, 173], [100, 0]))

        left, right = leftHandAngle, rightHandAngle

        if left >= 70 and right >= 70:
            if direction == 0:
                counter += 0.5
                rep_start_time = time.time()  # Start the timer when a new rep starts
                direction = 1
        if left <= 70 and right <= 70:
            if direction == 1:
                counter += 0.5
                direction = 0
                rep_duration = time.time() - rep_start_time  # Calculate the duration of the completed rep
                print(f"Time for rep {counter}: {round(rep_duration,3)}")
                repSpeedArray.append(round(rep_duration,3))

        cv2.rectangle(img, (0, 0), (120, 120), (255, 0, 0), -1)
        cv2.putText(img, str(int(counter)), (20, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.6, (0, 0, 255), 7)

        leftval = np.interp(left, [0, 100], [400, 200])
        rightval = np.interp(right, [0, 100], [400, 200])

        if left > 70:
            cv2.rectangle(img, (952, int(leftval)), (995, 400), (0, 0, 255), -1)

        if right > 70:
            cv2.rectangle(img, (8, int(rightval)), (50, 400), (0, 0, 255), -1)


while True:
    ret, img = cap.read()
    if not ret:
        break

    #img = cv2.resize(img, (1000, 500))
    cvzone.putTextRect(img, 'AI Push Up Counter', [345, 30], thickness=2, border=2, scale=2.5)
    pd.findPose(img, draw=False)
    lmlist, bbox = pd.findPosition(img, draw=False, bboxWithHands=False)

    angles(lmlist, 11, 13, 15, 12, 14, 16, drawpoints=True)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nRep Speed Array: ",repSpeedArray)

failurerep=failureRep(repSpeedArray)

rating = 10 - (failurerep-len(repSpeedArray))

print(f"RATING : {rating}")
