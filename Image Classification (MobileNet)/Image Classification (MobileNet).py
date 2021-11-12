# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:23:02 2021
@author: Eriny
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import warnings
warnings.simplefilter('ignore')


## Download MobileNet Pre-trained Model
trainedModel = tf.keras.applications.mobilenet.MobileNet()
trainedModel.summary()

def prepareImages(imageName):
    imgPath = r'./images/'
    ## Preprocess The Image
    img = image.load_img(imgPath+imageName, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    return tf.keras.applications.mobilenet.preprocess_input(img)


#Image(filename='/images/frog_612x612_before_preprocessing.jpg', width=612, height=612)

imageName = 'frog_612x612_before_preprocessing.jpg'
imageName = 'coffee.jpg'
imageName = 'strawberry.jpg'
imageName = 'face.jpg' ## Very Bad Prediction --> MobileNet Hadn't Be Trained On This Type 
imageName = 'strawberry2.jpg'
img = prepareImages(imageName)
predictions = trainedModel.predict(img)
#print(predictions)
## Probabilities For Top 5 Predictions Of The Input Image
predictions = imagenet_utils.decode_predictions(predictions)
print(predictions)


