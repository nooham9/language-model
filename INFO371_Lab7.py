#!/usr/bin/env python3
##
import numpy as np
import pandas as pd
import tensorflow as tf
print("tensorflow version", tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import os

## Data download: https://www.kaggle.com/c/dogs-vs-cats/data
## Location of images.  You should have two folder in this place:
## 'train' and 'test'
imgDir = "../../data"
## resize images to this dimension below
imageWidth, imageHeight = 256, 256
imageSize = (imageWidth, imageHeight)
imgChannels=3

## define other constants, including command line argument defaults
epochs = 1

## Prepare dataset for training model:
filenames = os.listdir(os.chdir("./data/mini_train"))

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

print(df.sample(8))

print("Training on", df.shape[0], "images")
print("categories:\n", df.category.value_counts())
# categories are the 5 languages

## Create model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization

# sequential (not recursive) model (one input, one output)
model=Sequential()

model.add(Conv2D(48, 3, activation='relu',
                 input_shape=(imageWidth, imageHeight, imgChannels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(96, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(192, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

## Training data generator:
trainGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
).\
flow_from_dataframe(
    df,
    os.path.join(imgDir, "train"),
    x_col='filename', y_col='category',
    target_size=imageSize,
    class_mode='categorical')

## Model Training:
history = model.fit(
    trainGenerator,
    epochs=epochs
)

##  Test your model performance on custom data
from PIL import Image
import numpy as np
##  feel free to add your images here, remove others!
## pick 20 random file names from testing dir
testFNames = np.random.choice(os.listdir(os.chdir("./data/train-cropped")), size=20)
## add "test/" to these file names
testFNames = ["train/" + f for f in testFNames]

####END OF CURR CHANGES

## replace first of these with our own images
print("Plot example results")
labelMap = {0:"cat", 1:"dog"}
plt.figure(figsize=(12, 24))
index = 1
rows, cols = 4, 5
for testFName in testFNames:
    img = load_img(os.path.join(imgDir, testFNames[index-1]), target_size=imageSize)
    plt.subplot(rows, cols, index)
    plt.imshow(img)
    im = Image.open(os.path.join(imgDir, testFName))
    im = im.resize(imageSize)
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    im = im/255
    phat = model.predict([im])
    pred = np.argmax(phat, axis=-1)[0]
    print("sample image {} [{}] is {}".format(testFName, phat, labelMap[pred]))
    plt.xlabel("{} ({:.3})".format(labelMap[pred], phat.max()))
    if index >= len(testFNames):
        break
    index += 1
plt.tight_layout()
plt.show()
