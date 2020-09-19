#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2018 Created by Yiming Peng and Bing Xue
"""
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse

import numpy as np
import random
import tensorflow as tf
import sys

from conf_mtrx_plt import plot_confusion_matrix

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args.add_argument("pretrainedNN", default = "ResNet50",
                      help = "pretrainedNN")
    args.add_argument("n", default = "5",
                      help = "n iteration num")
    args = vars(args.parse_args())
    return args


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


def evaluate(X_test, y_test, pretrainedNN, n):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model("model/model_"+ pretrainedNN + "_" + str(n) +".h5")
    evs=model.evaluate(X_test, y_test, batch_size, verbose = 1)
    prds=model.predict(X_test)
    return evs, prds


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    
    print('sys.argv', sys.argv)

    #pretrainedNN=sys.argv[1]
    #n=sys.argv[2]

    # Test folder
    test_data_dir = args["test_data_dir"]
    pretrainedNN=args["pretrainedNN"]
    n=args["n"]

    # Image size, please define according to your settings when training your model.
    image_size = (32, 32)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    X_test = preprocess_data(X_test)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    evs, prds = evaluate(X_test, y_test, pretrainedNN, n)
    #print("evs",evs)
    #print(type(y_test), prds.shape)
    max_index_prds = np.argmax(prds, axis=1)
    #print(max_index_prds)
    max_index_ys = np.argmax(y_test, axis=1)
    #print(max_index_ys)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true=max_index_ys, y_pred=max_index_prds)

    #cm = tf.math.confusion_matrix(labels=max_index_ys, predictions=max_index_prds)    
    print("Confusion matrixion_matrix", cm)
    
    print("loos, accuracy", evs)
    
    plot_confusion_matrix(cm,range(77))


    #print("loss={}, accuracy={}".format(loss, accuracy))
