# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:16:21 2021
@author: Eriny
"""

import os
import numpy as np
import cv2
import Functions as fun

### Collect The Dataset & Keypoint Values For Training and Testing

## Create A Folder For Each Sign Word
for action in fun.SIGNS_WORDS: 
    ## Create 30 Folders. 
    for video_no in range(fun.NO_OF_FEAMES):
        try: 
            os.makedirs(os.path.join(fun.DATA_PATH, action, '0'+str(video_no)))
        except:
            pass

## Start Collecting Data
cap = cv2.VideoCapture(0)
with fun.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    for actionSign in fun.SIGNS_WORDS:
        for vedioNo in range(fun.NO_OF_VEDIOS):
            for frameNo in range(fun.NO_OF_FEAMES):

                _, frame = cap.read()

                ## Make Metections
                image, results = fun.detectMediapipe(frame, holistic)

                ## Draw Landmarks
                fun.drawLandmarks(image, results)
                
                ## 
                if frameNo == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting Frames For {} Video Number {}'.format(actionSign, vedioNo), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting Frames For {} Video Number {}'.format(actionSign, vedioNo), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                ## Export Keypoints --> Numpy Array
                keypoints = fun.extractKeypoints(results)
                npy_path = os.path.join(fun.DATA_PATH, actionSign, '0'+str(vedioNo), str(frameNo))
                np.save(npy_path, keypoints)

                key = cv2.waitKey(1)
                if key == 27:
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

