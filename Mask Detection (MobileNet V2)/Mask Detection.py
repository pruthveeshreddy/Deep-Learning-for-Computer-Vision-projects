# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:31:12 2021
@author: Eriny
"""

import os 
import shutil
import random
import seaborn as sns

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten, Dense, AveragePooling2D, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy
from tensorflow.keras.models import Model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import warnings
warnings.simplefilter('ignore')

# initialize the initial learning rate, number of epochs to train for,
# and batch size
LEARNING_RATE      = 1e-4 ## 1e-4 = 0.0001
NO_ON_EPOCHS       = 2  
BATCH_SIZE         = 32   ## For SGD
DIRECTORY          = r".\dataset"
CATEGORIES         = ["with_mask", "without_mask"]
NO_OF_CLASSES      = 2
TRAININGSET_PATH   = r'./dataset/training'
VALIDATIONSET_PATH = r'./dataset/validation'
TESTINGSET_PATH    = r'./dataset/testing'


## Load & Split Downloaded Github Data Into Training, Validation and Testing Dirs.
os.chdir(r"./dataset/")
print("\n\nLoading Images...")

if os.path.isdir('training/with_mask/') is False:
    print('???', os.path.isdir('training/with_mask/') is False )
    os.mkdir('training')
    os.mkdir('validation')
    os.mkdir('testing')
    
    for category in CATEGORIES:
        shutil.move(f'{category}', 'training') ## Move Whole(2) Data Folders To (training) Folder
        os.mkdir(f'validation/{category}')     ## Create Empty Data Folders In (validation) Folder
        os.mkdir(f'testing/{category}')        ## Create Empty Data Folders In (testing) Folder
        
        ## Move Random 383 Item (Image) From Each Folder In (training) Folder To (validation) Folder
        validationSamples = random.sample(os.listdir(f'training/{category}'), 383)
        for j in validationSamples:
            shutil.move(f'training/{category}/{j}', f'validation/{category}')

        ## Move Random 191 Item (Image) From Each Folder In (training) Folder To (testing) Folder
        testingSamples = random.sample(os.listdir(f'training/{category}'), 191)
        for k in testingSamples:
            shutil.move(f'training/{category}/{k}', f'testing/{category}')

os.chdir('../')
#print(os.getcwd())

## Construct The Training Image Generator For Data Augmentation
training_batches = ImageDataGenerator(
                                        preprocessing_function=preprocess_input,
                                        rotation_range=20,  zoom_range=0.15, 
                                        height_shift_range=0.2,
                                        shear_range=0.15, fill_mode="nearest"
                                     ).flow_from_directory(directory=TRAININGSET_PATH, 
                                                           target_size=(224,224), batch_size=10)
print("training_batches:",len(training_batches))
validation_batches = ImageDataGenerator(
                                        preprocessing_function=preprocess_input,
                                        rotation_range=20,  zoom_range=0.15, 
                                        height_shift_range=0.2,
                                        shear_range=0.15, fill_mode="nearest"
                                     ).flow_from_directory(directory=VALIDATIONSET_PATH, 
                                                           target_size=(224,224), batch_size=10)
                                                          
testing_batches = ImageDataGenerator(
                                        preprocessing_function=preprocess_input,
                                        rotation_range=20,  zoom_range=0.15, 
                                        height_shift_range=0.2,
                                        shear_range=0.15, fill_mode="nearest"
                                     ).flow_from_directory(directory=TESTINGSET_PATH, 
                                                           target_size=(224,224), batch_size=10, 
                                                           shuffle=False)

                                                           
## Perform One-Hot Encoding On The Labels/Classes
oneHotEncoder = LabelBinarizer()
CATEGORIES    = oneHotEncoder.fit_transform(CATEGORIES) ## OUTPUT: Vector mx1 (m: Number Of Images)
CATEGORIES    = to_categorical(CATEGORIES)          ## OUTPUT: Matrix nxm (n: Number Of Classes)

                                                           
## Load The MobileNetV2 network, ensuring the head FC layer sets are left off
baseModel = MobileNetV2(include_top=False, ## include_top=False: Remove Output Layer
	                    input_tensor=Input(shape=(224, 224, 3)))


## The Output Layer & Another Added Layers
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(NO_OF_CLASSES, activation="softmax")(headModel)

## Create The Model
maskDetectModel = Model(inputs=baseModel.input, outputs=headModel)
maskDetectModel.summary()

## No Updates For The Weight Of All Layers In First Backbropagation Process 
for layer in baseModel.layers:
	layer.trainable = False


## Compile The Model
print("\n\nCompiling The Model...")
opt = Adam(lr=LEARNING_RATE, 
           decay=LEARNING_RATE / NO_ON_EPOCHS)

maskDetectModel.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])


### Tain The Model
print("\n\nTraining The Model...")
maskDetectModel.fit(x=training_batches, batch_size=BATCH_SIZE,
	                steps_per_epoch=len(training_batches),
	                validation_data=(validation_batches),
	                validation_steps=len(validation_batches),
	                epochs=NO_ON_EPOCHS, verbose=1)


### Test The Model (Using Testing Set --> New Data)
print("\n\nEvaluating The Model...")
predictions = maskDetectModel.predict(x=testing_batches, batch_size=BATCH_SIZE, verbose=0)
predictions = predictions.argmax(axis=1)

## Classification Report & Confusion Matrix & Heat Map Confusion Matrix
y_test = testing_batches.classes
report = classification_report(y_test, predictions, target_names=oneHotEncoder.classes_)
cm     = confusion_matrix(y_true=y_test, y_pred=predictions)

print('Classification Report:\n',report)
print('Confusion Matrix:\n',cm)
sns.heatmap(cm, annot=True, cmap="BuPu")

### Save The Model To The Disk
print("Saving The Model...")
maskDetectModel.save("mask_detector.model", save_format="h5")