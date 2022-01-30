# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:43:52 2021
@author: Eriny
"""

import numpy as np
import random
import os
import shutil

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import TensorBoard

import Functions as fun

NO_OF_CLASSES      = fun.SIGNS_WORDS.shape[0]
BATCH_SIZE         = 10
NO_ON_EPOCHS       = 1000
TRAININGSET_PATH   = r'./MP_Data/training'
VALIDATIONSET_PATH = r'./MP_Data/validation'
TESTINGSET_PATH    = r'./MP_Data/testing'
VALIDATIONSET_SIZE = int(fun.NO_OF_VEDIOS * 20 / 100)
TESTINGSET_SIZE    = int(fun.NO_OF_VEDIOS * 10 / 100)
TRAININGSET_SIZE   = fun.NO_OF_VEDIOS - (VALIDATIONSET_SIZE + TESTINGSET_SIZE)

## Load & Split Collected Data Into Training, Validation and Testing Dirs.
os.chdir(r"./MP_Data/")
print("\n\nLoading Images...")

if os.path.isdir('training/') is False:
    print('???', os.path.isdir('training/') is False )
    os.mkdir('training')
    os.mkdir('validation')
    os.mkdir('testing')
    
    for word in fun.SIGNS_WORDS:
        shutil.move(f'{word}', 'training') ## Move Whole Data Folders To (training) Folder
        os.mkdir(f'validation/{word}')     ## Create Empty Data Folders In (validation) Folder
        os.mkdir(f'testing/{word}')        ## Create Empty Data Folders In (testing) Folder
        
        ## Move Random 20% Items (Image) From Each Folder In (training) Folder To (validation) Folder
        validationSamples = random.sample(os.listdir(f'training/{word}'), VALIDATIONSET_SIZE)
        i = 0
        for j in validationSamples:
            os.rename(f'training/{word}/{j}', f'training/{word}/{i}')
            shutil.move(f'training/{word}/{i}', f'validation/{word}')
            i += 1

        ## Move Random 20% Items (Image) From Each Folder In (training) Folder To (testing) Folder
        testingSamples = random.sample(os.listdir(f'training/{word}'), TESTINGSET_SIZE)
        i = 0
        for k in testingSamples:
            os.rename(f'training/{word}/{k}', f'training/{word}/{i}')
            shutil.move(f'training/{word}/{i}', f'testing/{word}')
            i += 1
        
        ## Rename 
        i = 0
        for r in os.listdir(f'training/{word}'):
            os.rename(f'training/{word}/{r}', f'training/{word}/{i}')
            i += 1

os.chdir('../')
#print(os.getcwd())


## Encode The Labels/Classes
label_map = {label:num for num, label in enumerate(fun.SIGNS_WORDS)}
#print(label_map)

## Get Whole Keypoints In One Array & All Labels In One Array
sequences, labels = [], []
for action in fun.SIGNS_WORDS:
    for videoNo in range(TRAININGSET_SIZE):
        window = [] ## Whole Frames For Each Video For Each Sign/Action
        for frameNo in range(fun.NO_OF_FEAMES):
            res = np.load(os.path.join(fun.DATA_PATH+'/training', action, str(videoNo), "{}.npy".format(frameNo)))
            window.append(res)
        sequences.append(window) ## Array Include An Array That Include Whole Frames For Each Video For Each Sign/Action
        labels.append(label_map[action])

X_train = np.array(sequences)
Y_train = np.array(labels)


## Get Whole Keypoints In One Array & All Labels In One Array
sequences, labels = [], []
for action in fun.SIGNS_WORDS:
    for videoNo in range(VALIDATIONSET_SIZE):
        window = [] ## Whole Frames For Each Video For Each Sign/Action
        for frameNo in range(fun.NO_OF_FEAMES):
            res = np.load(os.path.join(fun.DATA_PATH+'/validation', action, str(videoNo), "{}.npy".format(frameNo)))
            window.append(res)
        sequences.append(window) ## Array Include An Array That Include Whole Frames For Each Video For Each Sign/Action
        labels.append(label_map[action])

X_valid = np.array(sequences)
Y_valid = np.array(labels)


## Get Whole Keypoints In One Array & All Labels In One Array
sequences, labels = [], []
for action in fun.SIGNS_WORDS:
    for videoNo in range(TESTINGSET_SIZE):
        window = [] ## Whole Frames For Each Video For Each Sign/Action
        for frameNo in range(fun.NO_OF_FEAMES):
            res = np.load(os.path.join(fun.DATA_PATH+'/testing', action, str(videoNo), "{}.npy".format(frameNo)))
            window.append(res)
        sequences.append(window) ## Array Include An Array That Include Whole Frames For Each Video For Each Sign/Action
        labels.append(label_map[action])

X_test = np.array(sequences)
Y_test = np.array(labels)


## One-Hot Encode The Labels/Classes
Y_train = to_categorical(Y_train).astype(int)
Y_valid = to_categorical(Y_valid).astype(int)
Y_test = to_categorical(Y_test).astype(int)


### Build LSTM Neural Network

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(NO_OF_CLASSES, activation='softmax'))

## Compile The Model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

## Train The Model
model.fit(X_train, Y_train,
 	                validation_data=(X_valid, Y_valid),
 	                epochs=NO_ON_EPOCHS, verbose=1)

## Make Predictions

