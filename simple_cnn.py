import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import glob
from PIL import Image

PATH_TO_DATASETS = ['./DROWSY/**/5', './FOCUSED/**/5', 'UNFOCUSED/**/5']

def handle_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
            help="path to input dataset of images")
    ap.add_argument("-m", "--model", required=True,
            help="path to output trained model")
    ap.add_argument("-l", "--label-bin", required=True,
            help="path to output label binarizer")
    ap.add_argument("-p", "--plot", required=True,
            help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())

    return args


def determine_data_paths(path_to_datasets):
    data = []
    labels = []

    for path_to_dataset in path_to_datasets:
        dirs = glob.glob(path_to_dataset, recursive=True)
        image_paths = sorted(list(dirs))

        # image paths is the 3 eeg_record experiments that we are going to read in
        image_paths = image_paths[:3]

        # need to read in all images for each eeg_record experiment
        images = []
        for image_path in image_paths:
            png_path = '{0}/*.png'.format(image_path)
            curr_images = glob.glob(png_path, recursive=True)
            images.extend(curr_images)

        for png in images:
            image_src = np.asarray(Image.open(png).resize((32, 32)).convert('RGB'))
            #image_src = cv2.resize(image_src, (32, 32)).flatten()
            data.append(image_src)
            label = path_to_dataset.split(os.path.sep)[-3]
            labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    print('data length: {0}'.format(len(data)))
    print('labels length: {0}'.format(len(labels)))

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    print('Train length: {0}'.format(len(trainX)))
    print('Test length: {0}'.format(len(testY)))

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(64, activation='relu'))

    # using softmax becuase we are expecting more than two outcomes
    classifier.add(Dense(3, activation='softmax'))

    classifier.summary()

    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.01
    EPOCHS = 75
    # compile the model using SGD as our optimizer and categorical
    # cross-entropy loss (you'll want to use binary_crossentropy
    # for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(lr=INIT_LR)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    # train
    history = classifier.fit(trainX, trainY, epochs=10, validation_data=(testX, testY), batch_size=32)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = classifier.evaluate(testX, testY, verbose=2)

    print(test_acc)

if __name__ == '__main__':
    handle_args()

    determine_data_paths(PATH_TO_DATASETS)
