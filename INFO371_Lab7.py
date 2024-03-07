#!/usr/bin/env python3
##
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

## Location of images.  You should have two folder in this place:
## 'train' and 'test'
imgDir = "../../data"
## resize images to this dimension below
imageWidth, imageHeight = 256, 256
imageSize = (imageWidth, imageHeight)
imgChannels= 1

## define other constants, including command line argument defaults
epochs = 1

## Prepare dataset for training model:
filenames = os.listdir(os.chdir("./data/train"))

print(len(filenames), "images found")

# categories are the 5 languages
languages = ["EN", "RU", "ZN", "DA", "TH"]

def match_lang(filename, languages):
    for language in languages:
        if language in filename:
            return language
        
df = pd.DataFrame({
    'filename':filenames,
    'category': [match_lang(filename, languages) for filename in filenames]
})
# ADJUST WHEN CHANGING LANGUAGES
# limiting to just EN and ZN
# df = df[df['category'].isin(['EN', 'ZN'])]

print(df.sample(8))

print("Training on", df.shape[0], "images")
print("categories:\n", df.category.value_counts())
# categories are the 5 languages

## Create model
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization

# sequential (not recursive) model (one input, one output)
model=Sequential()

model.add(Conv2D(48,
                 kernel_size= 3,
                 strides = 1,
                 activation='relu',
                 kernel_initializer = initializers.HeNormal(),
                 input_shape=(imageWidth, imageHeight, imgChannels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(96,
                 kernel_size= 3,
                 kernel_initializer = initializers.HeNormal(),

                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(192, 
                 kernel_size= 3,
                 kernel_initializer = initializers.HeNormal(),

                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# ADJUST # WHEN CHANGING LANGUAGES
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

## Training data generator:
trainGenerator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
).\
flow_from_dataframe(
    df,
    os.path.join(imgDir, "train"),
    x_col='filename', y_col='category',
    target_size=imageSize,
    class_mode='categorical',
    color_mode="grayscale",
    shuffle=True
)

## Model Training:
history = model.fit(
    trainGenerator,
    epochs=epochs
)

# ADJUST WHEN CHANGING LANGUAGES
# testDir = os.path.join(imgDir, "EN-ZN")

testDir = os.path.join(imgDir, "test")
fNames = os.listdir(testDir)
dfTest = pd.DataFrame({
    'filename': fNames,
    'category': [match_lang(filename, languages) for filename in fNames]
})
print(dfTest.shape, "test files read from", testDir)

test_generator = ImageDataGenerator(
    rescale=1./255
    # do not randomize testing!
).flow_from_dataframe(
    dfTest,
    # os.path.join(imgDir, "EN-ZN"),
    testDir,
    x_col='filename',
    class_mode = None,  # we don't want target for prediction
    target_size = imageSize,
    shuffle = False,
    # do _not_ randomize the order!
    # this would clash with the file name order!
    color_mode="grayscale"
)
phat = model.predict(test_generator)
print(phat)
dfTest['category'] = np.argmax(phat, axis=-1)
print(dfTest['category'])

# ADJUST WHEN CHANGING LANGUAGES
label_map = {0:"EN", 1:"RU", 2:"ZN", 3:"DA", 4:"TH"}
dfTest['category'] = dfTest['category'].replace(label_map)
print("post mapping")
print(dfTest['category'].value_counts())


