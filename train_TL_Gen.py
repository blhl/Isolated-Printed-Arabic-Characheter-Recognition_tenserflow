#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import backend as K
#temp chg AB
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.python.keras.applications.resnet import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16

import keras

from imutils import paths

from sklearn.preprocessing import LabelBinarizer

import cv2
import os
import argparse
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator
import sys

import numpy as np
import tensorflow as tf
import random

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def construct_model(pretrainedNN):

    model = Sequential()
    if(pretrainedNN=='VGG16'):
        model.add(VGG16(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='VGG19'):
        model.add(VGG19(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='ResNet101'):
        model.add(ResNet101(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='ResNet152'):
        model.add(ResNet152(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='ResNet50V2'):
        model.add(ResNet50V2(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='ResNet101V2'):
        model.add(ResNet101V2(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='ResNet152V2'):
        model.add(ResNet152V2(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='MobileNet'):
        model.add(MobileNet(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='MobileNetV2'):
        model.add(MobileNetV2(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='DenseNet121'):
        model.add(DenseNet121(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='DenseNet169'):
        model.add(DenseNet169(weights= None, include_top=False, input_shape= (32,32,3)))
    elif(pretrainedNN=='DenseNet201'):
        model.add(DenseNet201(weights= None, include_top=False, input_shape= (32,32,3)))
    else:
        model.add(ResNet50(weights= None, include_top=False, input_shape= (32,32,3)))
   
    model.add(Flatten())

    model.add(Dense(77, activation='softmax'))
   
    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
    return model

def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype = "float") / 255.0
    y_test = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return X


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)



def train_model(model, n):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """
#    args = parse_args()

    # Test folder
#    train_data_dir = args["train_data_dir"]
    train_data_dir='data/Train_data'
    # Image size, please define according to your settings when training your model.
    image_size = (32, 32)

    # Load images
    images, labels = load_images(train_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_train, y_train = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    X_train = preprocess_data(X_train)
    print("\nI m before fit.")
    model.fit(X_train, y_train, batch_size=20, epochs=int(n), verbose=1, validation_split=0.1)

#    model.fit(X_train, y_train, batch_size=20, epochs=10, verbose=1, 
#                   validation_data=(X_train, y_train))
    print("\nI m after fit.")
    #model.evaluate(X_train, y_train)
    # Add your code here
    return model

'''/////////////
model1 = createModel()
batch_size = 256
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))
 
model1.evaluate(test_data, test_labels_one_hot)

#-------------
'''


def save_model(model, pretrainedNN, n):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    model.save("model/model_"+ pretrainedNN + "_" + str(n) +".h5")
    print("Model Saved Successfully.")


if __name__ == '__main__':
    pretrainedNN=sys.argv[1]
    n=sys.argv[2]
    #print (type(n))
    model = construct_model(pretrainedNN)
    model = train_model(model, n)
    save_model(model, pretrainedNN, n)
