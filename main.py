from imutils.video import VideoStream
from imutils import face_utils
# import argparse
import imutils
import dlib
import cv2 as cv
# from operator import itemgetter
import numpy as np
import itertools
import math


def roundness(x):
    convex_hull = cv.convexHull(x)
    return cv.arcLength(convex_hull, True) ** 2 / (4 * math.pi * cv.contourArea(convex_hull))

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# get the indexes of the landmarks for the left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("Camera starting up...")
vs = VideoStream().start()

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    frame = cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame using dlib frontal face detector
    faces = face_detector(gray, 0)

    # loop over the face detections
    for face in faces:
        # find the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = shape_predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # get left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        roi_left = cv.GaussianBlur(gray[leftEye[1][1]:leftEye[4][1], leftEye[0][0]:leftEye[3][0]], (7, 7), 0)
        roi_right = cv.GaussianBlur(gray[rightEye[1][1]:rightEye[4][1], rightEye[0][0]:rightEye[3][0]], (7, 7), 0)

        # _, roi_left = cv.threshold(roi_left, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        _, roi_left = cv.threshold(roi_left, 100, 255, cv.THRESH_BINARY_INV)
        # _, roi_right = cv.threshold(roi_right, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        _, roi_right = cv.threshold(roi_right, 100, 255, cv.THRESH_BINARY_INV)
        contours_left, _ = cv.findContours(roi_left, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # print(_)
        contours_right, _ = cv.findContours(roi_right, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        contours_left = sorted(contours_left, key=lambda x: cv.contourArea(x), reverse=True)
        # contours_left = sorted(contours_left, key=lambda x: cv.arcLength(cv.convexHull(x), True) ** 2 / (4*math.pi*cv.contourArea(cv.convexHull(x))), reverse=False)
        contours_right = sorted(contours_right, key=lambda x: cv.contourArea(x), reverse=True)
        # contours_right = sorted(contours_right, key=lambda x: cv.arcLength(cv.convexHull(x),
        #                                                                  True) ** 2 / (4 * math.pi * cv.contourArea(cv.convexHull(x))), reverse=False)

        for contour in contours_left:
            contour = cv.convexHull(contour)
            # area = cv.contourArea(contour)
            # if area == 0:
            #     continue
            # circumference = cv.arcLength(contour, True)
            # circularity = circumference ** 2 / (4 * math.pi * area)

            for i in contour:
                # print("I: ", i)
                i[0][0] += leftEye[0][0]
                i[0][1] += leftEye[1][1]

            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

            # if circularity < 1.5:
            # cv.drawContours(frame, [contour], -1, (0,255,0), 2)
            break

        for contour in contours_right:
            contour = cv.convexHull(contour)
            # area = cv.contourArea(contour)
            # if area == 0:
            #     continue
            # circumference = cv.arcLength(contour, True)
            # circularity = circumference ** 2 / (4 * math.pi * area)

            for i in contour:
                # print("I: ", i)
                i[0][0] += rightEye[0][0]
                i[0][1] += rightEye[1][1]

            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

            # if circularity < 1.5:
            # cv.drawContours(frame, [contour], -1, (0,255,0), 2)
            break

    cv.imshow("Camera", frame)
    # if the `d` key was pressed, break from the loop
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

cv.destroyAllWindows()
vs.stop()
