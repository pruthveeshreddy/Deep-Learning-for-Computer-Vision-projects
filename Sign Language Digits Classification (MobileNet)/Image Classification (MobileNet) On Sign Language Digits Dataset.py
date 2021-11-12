# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:52:48 2021
@author: Eriny
"""


import os
import shutil
import random
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter('ignore')

NO_OF_CLASSES = 10
NO_OF_EPOCHES = 5 # 30
os.chdir(r"./Sign-Language-Digits-Dataset-master/Dataset/")


## Organize Downloaded Github Data Into Training, Validation and Testing Dirs.
if os.path.isdir('training/0/') is False:
    os.mkdir('training')
    os.mkdir('validation')
    os.mkdir('testing')
    
    for i in range(0,10):
        shutil.move(f'{i}', 'training') ## Move Whole(10) Data Folders To (training) Folder
        os.mkdir(f'validation/{i}')     ## Create Empty Data Folders In (validation) Folder
        os.mkdir(f'testing/{i}')        ## ## Create Empty Data Folders In (testing) Folder
        
        ## Move Random 30 Item (Image) From Each Folder In (training) Folder To (validation) Folder
        validationSamples = random.sample(os.listdir(f'training/{i}'), 30)
        for j in validationSamples:
            shutil.move(f'training/{i}/{j}', f'validation/{i}')

        ## Move Random 5 Item (Image) From Each Folder In (training) Folder To (testing) Folder
        testingSamples = random.sample(os.listdir(f'training/{i}'), 5)
        for k in testingSamples:
            shutil.move(f'training/{i}/{k}', f'testing/{i}')

os.chdir('../..')

trainingsetPath   = r'./Sign-Language-Digits-Dataset-master/Dataset/training'
validationsetPath = r'./Sign-Language-Digits-Dataset-master/Dataset/validation'
testingsetPath    = r'./Sign-Language-Digits-Dataset-master/Dataset/testing'

training_batches = ImageDataGenerator(
                                        preprocessing_function=keras.applications.mobilenet.preprocess_input
                                     ).flow_from_directory(directory=trainingsetPath, 
                                                           target_size=(224,224), batch_size=10)
validation_batches = ImageDataGenerator(
                                        preprocessing_function=keras.applications.mobilenet.preprocess_input
                                     ).flow_from_directory(directory=validationsetPath, 
                                                           target_size=(224,224), batch_size=10)
                                                           
testing_batches = ImageDataGenerator(
                                        preprocessing_function=keras.applications.mobilenet.preprocess_input
                                     ).flow_from_directory(directory=testingsetPath, 
                                                           target_size=(224,224), batch_size=10, 
                                                           shuffle=False)
                                                           
## Download MobileNet Pre-trained Model
trainedModel = tf.keras.applications.mobilenet.MobileNet()
#trainedModel.summary()
                                                           
### Create Out Model From Pre-Trained Model   
        
## Remove Last 5 Layers From MobileNet Model. [INPUT LAYER]
x = trainedModel.layers[-6].output
#print(x)

## The Output Layer.                          [OUTPUT LAYER]
output = Dense(units=NO_OF_CLASSES, activation='softmax')(x)
                   
## Create The Model
signLangModel = Model(inputs=trainedModel.input, outputs=output)
#signLangModel.summary()

## No Updates For The Weight Of Some Layers In First Backbropagation Process
for layer in signLangModel.layers[:-23]:
    layer.trainable = False
#signLangModel.summary()


## Compile The Model
signLangModel.compile(optimizer=Adam(lr=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

### Tain The Model
signLangModel.fit(x=training_batches, steps_per_epoch=len(training_batches), ## No Batch Size
                  validation_data=validation_batches, validation_steps=len(validation_batches),
                  epochs=NO_OF_EPOCHES, verbose=1)


### Test The Model (Using Testing Set --> New Data)
predictions = signLangModel.predict(x=testing_batches, steps=len(testing_batches), verbose=0)
y_test = testing_batches.classes

cm = confusion_matrix(y_true=y_test, y_pred=predictions.argmax(axis=1))
print(cm)

sns.heatmap(cm, annot=True, cmap="Blues")