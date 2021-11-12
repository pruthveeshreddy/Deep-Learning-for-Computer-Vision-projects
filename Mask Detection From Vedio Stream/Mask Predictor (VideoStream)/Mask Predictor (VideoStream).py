# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:23:59 2021
@author: Eriny
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from imutils.video import VideoStream

import numpy as np
import imutils
import cv2

prototxtPath = r"./Face Detector/deploy.prototxt"
weightsPath = r"./Face Detector/res10_300x300_ssd_iter_140000.caffemodel"

BATCH_SIZE = 32

def _detectTheFace(frame, faceNet):
    ## Pre-processing For Frames #frame, scale factor, frame size, mean --> https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224), (104.0, 177.0, 123.0))
    
    ## Obtain/Detect The Face From The Frames
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print('Shape:', detections.shape)

    return detections  

def predictMask(frame, faceNet, maskNet):

    detections = _detectTheFace(frame, faceNet)

    h, w = frame.shape[:2]
        
    faces = []
    locations = []
    predictions = []
    
    ## Loop Over The Detections
    for i in range(0, detections.shape[2]):
        ## Extract The Probability Associated With The Detection
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            ## Ensure That The Bounding Boxes Fall Within The Dimensions Of The Frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]       ## Extract The Face ROI
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) ## Convert It --> BGR To RGB 
            face = cv2.resize(face, (224, 224))          ## Resize It To Fit MobileNet Model --> 224x224
            face = img_to_array(face)
            face = preprocess_input(face)                ## preprocess it To Fit MobileNet Model --> 0:255 To -1:1 

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    ## At Least One Face Was Detected -->  Make A Predictions
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=BATCH_SIZE)

    ## Return A 2-Tuple Of Face Locations & Their Corresponding Locations
    return (locations, predictions)



## Load Pre-trained Face Detector Model
faceDetector = cv2.dnn.readNet(prototxtPath, weightsPath)

## Load The Mask Detector Model
maskDetectorModel = load_model("./Mask Predictor/mask_detector.model")

## Initialize The Video Stream
print("\n\nStarting Video Stream...")
vedioStream = VideoStream(src=0).start()


# loop over the frames from the video stream
while True:

    frame = vedioStream.read()
    frame = imutils.resize(frame, width=400)

    ## Detect Faces & Predict Is Masked Or Not
    (locations, isMasked) = predictMask(frame, faceDetector, maskDetectorModel)

    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locations, isMasked):
        ## Unpack The Bounding Box & Predictions
        (startX, startY, endX, endY) = box
        (withMask, withoutMask)      = pred

        ## Preparing Drawed Triangle 
        label = "Mask" if withMask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(withMask, withoutMask)*100)
        ## Display It
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    ## Show The Output Frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

## Cleanup
vedioStream.stop()
cv2.destroyAllWindows()