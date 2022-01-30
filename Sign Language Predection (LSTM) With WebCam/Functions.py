import cv2
import mediapipe as mp
import numpy as np
import os

### 
mp_holistic  = mp.solutions.holistic                     ## Holistic Model    --> Make Detections
mp_drawing   = mp.solutions.drawing_utils                ## Drawing Utilities --> Draw These Detections
SIGNS_WORDS  = np.array(['hello', 'thanks', 'iloveyou']) ## Words Of Signs
DATA_PATH    = os.path.join('MP_Data')                   ## Path Of Exported Data & Numpy Arrays Of Frames
NO_OF_VEDIOS = 30                                        ## 30 Videos For Each Sign
NO_OF_FEAMES = 30                                        ## 30 Frames For A Video. [Each Video Folder Has 30 Frames/Numpy Array]

def detectMediapipe(image, holisticModel):
    ''' 
        Returns Lanmarks List & The Image/Frame Itself 
        
        • OpenCV Reads Imgs In BGR Channel Format, mediapipe Need Imgs In RGB Channel Format
    ''' 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = holisticModel.process(image)         ## Make Prediction 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, results

    
def drawLandmarks(image, results):
    ## Draw Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), ## Dot Color
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) ## Line Color
                             ) 
    ## Draw Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    ## Draw Left Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    ## Draw Right Hand Connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extractKeypoints(results):
    ''' 
        Extract keypoints values if there are. Set keypoints to zeros if aren't there.
        Returns Array Of Concatenated Arrays Of Pose, Face Left and Rights Hand Landmarks Points.
        
        132  = 4*len(results.right_hand_landmarks.landmark), 
        1404 = 3*len(results.right_hand_landmarks.landmark),
        63   = 3*len(results.right_hand_landmarks.landmark),
        63   = 3*len(results.right_hand_landmarks.landmark)
    '''
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh]) ## Shape: 33*4 + 468*3 + 21*3 + 21*3 --> (1662,)
    