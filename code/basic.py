import mediapipe as mp
import cv2 as cv
import numpy as np
import time
import os

cap=cv.VideoCapture(0)

mesh=mp.solutions.face_mesh
face = mesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
fin= mp.solutions.drawing_utils
spec=fin.DrawingSpec(thickness=1,circle_radius=1)

while True:
    ret,frame=cap.read()
        
    cvtd=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    cvtd = cv.flip(frame,1)

    processed= face.process(cvtd)
    if processed.multi_face_landmarks:
        for i in processed.multi_face_landmarks:
            fin.draw_landmarks(cvtd,i,mesh.FACEMESH_CONTOURS,spec,spec)
    cv.imshow('frame',cvtd)
    if cv.waitKey(1)==ord('q'):
        break
cv.destroyAllWindows
