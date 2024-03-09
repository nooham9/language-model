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
epochs =10 

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

model.add(Conv2D(32,
                 kernel_size= 6,
                 strides = 3,
                 activation='relu',
                 kernel_initializer = initializers.HeNormal(),
                 input_shape=(imageWidth, imageHeight, imgChannels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(64,
                 kernel_size= 6,
                 strides = 3,
                 kernel_initializer = initializers.HeNormal(),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(128,
                 kernel_size= 3,
                 strides = 1,
                 kernel_initializer = initializers.HeNormal(),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
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
languages = ["EN", "RU", "ZN", "DA", "TH"]


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
# ADJUST WHEN CHANGING LANGUAGES
languages = ["EN", "RU", "ZN", "DA", "TH"]

label_map = {0:"EN", 1:"RU", 2:"ZN", 3:"DA", 4:"TH"}
print("post mapping")
print(dfTest['category'].value_counts())


dfTest['predicted'] = pd.Series(np.argmax(phat, axis = -1))
print(dfTest.head())

dfTest["predicted"] = dfTest.predicted.replace(label_map)
print(dfTest['predicted'].value_counts())

print("confusion matrix (validation)")
print(pd.crosstab(dfTest.category, dfTest.predicted))
print("Validation accuracy", np.mean(dfTest.category == dfTest.predicted))

## Print and plot misclassified results
wrongResults = dfTest[dfTest.predicted != dfTest.category]
rows = np.random.choice(wrongResults.index, min(4, wrongResults.shape[0]), replace=False)
print("Example wrong results (validation data)")
print(wrongResults.sample(min(10, wrongResults.shape[0])))

## Plot 4 wrong and 4 correct results
plt.figure(figsize=(12, 12))
index = 1
for row in rows:
    filename = wrongResults.loc[row, 'filename']
    predicted = wrongResults.loc[row, 'predicted']
    img = load_img(os.path.join(imgDir, "test", filename), target_size=imageSize)
    plt.subplot(4, 2, index)
    plt.imshow(img)
    plt.xlabel(filename + " ({})".format(predicted))
    index += 1
# now show correct results
index = 5
correctResults = dfTest[dfTest.predicted == dfTest.category]
rows = np.random.choice(correctResults.index,
                        min(4, correctResults.shape[0]), replace=False)
for row in rows:
    filename = correctResults.loc[row, 'filename']
    predicted = correctResults.loc[row, 'predicted']
    img = load_img(os.path.join(imgDir, "test", filename), target_size=imageSize)
    plt.subplot(4, 2, index)
    plt.imshow(img)
    plt.xlabel(filename + " ({})".format(predicted))
    index += 1
plt.tight_layout()
plt.show()
