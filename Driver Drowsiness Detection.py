from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from playsound import playsound
import threading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def beep():
    threading.Thread(target=lambda: playsound("wakey.mp3")).start()

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

print("[INFO] initializing camera...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576

image_points = np.array([
    (359, 391), (399, 561), (337, 297),
    (513, 301), (345, 465), (453, 469)
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TIMER_STARTED = False
TIMER_START_TIME = 0
(mStart, mEnd) = (49, 68)

y_true = []
y_pred = []

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    rects = detector(gray, 0)

    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    for rect in rects:
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)

        # === Add manual ground truth (0 = awake, 1 = drowsy) ===
        actual_label = 1  # Change this based on your video
        y_true.append(actual_label)

        predicted_label = 0
        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            predicted_label = 1
        y_pred.append(predicted_label)

        if ear < EYE_AR_THRESH:
            if not TIMER_STARTED:
                TIMER_STARTED = True
                TIMER_START_TIME = time.time()
            elif time.time() - TIMER_START_TIME > 2:
                cv2.putText(frame, "DROWSINESS ALERT!", (500, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                beep()
        else:
            TIMER_STARTED = False

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0

        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (i, (x, y)) in enumerate(shape):
            if i in [33, 8, 36, 45, 48, 54]:
                index = [33, 8, 36, 45, 48, 54].index(i)
                image_points[index] = np.array([x, y], dtype='double')
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

# === Evaluate the model ===
if y_true and y_pred:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(conf_matrix)
else:
    print("[WARNING] y_true or y_pred is empty. No evaluation possible.")