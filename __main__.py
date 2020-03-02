import matplotlib
matplotlib.use('Agg')

# sklearn imports
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import glob
import random
import os

PATH_TO_DATASET = ['./DROWSY/**', './FOCUSED/**', 'UNFOCUSED/**']
ACTIVATION = 'relu'
PREDICT_ACTIVATION = 'softmax'
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']


def handle_arguments():
    """
    Function used to parse script arguments
    :return args: commandline arguments for script
    """
    """
    parser.add_argument("-d", "--dataset", required=True,
            help="path to input dataset of images")
    parser.add_argument("-m", "--model", required=True,
            help="path to output trained model")
    parser.add_argument("-l", "--label-bin", required=True,
            help="path to output label binarizer")
    parser.add_argument("-p", "--plot", required=True,
            help="path to output accuracy/loss plot")
    """

    parser = argparse.ArgumentParser(description='Train a model to classify spectrograms')
    parser.add_argument('-c', '--channel', dest='channel', type=int, required=True, choices=[4, 5, 8, 9, 10, 11, 16],
                        help='Flag used to determine what channel we want to create a model for')
    parser.add_argument('-s', '--set', dest='size', required=True,
                        help='Flag used to determine the amount of experiments to import for train/test data')
    parser.add_argument('-i', '--image-size', dest='image_size', required=True,
                        help='Flag used to determine the length and width to resize the data spectrogram images')
    parser.add_argument('-e', '--epochs', dest='epochs', required=True,
                        help='Flag used to determine the number of epochs for training')
    args = parser.parse_args()

    return args


def determine_data_paths(paths_to_datasets, channel, size):
    """
    Function used to determine full paths to datasets we will be reading for training and testing
    :param paths_to_datasets: path to root of data for DROWSY, FOCUSED, UNFOCUSED spectrograms
    :param channel: channel we want to train a model for
    :param size: number of experiments we want to input for test/train data
    :return:
    """
    all_paths = []

    for path_to_dataset in paths_to_datasets:
        path_with_channel = os.path.join(path_to_dataset, str(channel))
        dirs = glob.glob(path_with_channel, recursive=True)

        # TODO: need to figure out a way to get the same eeg_records dirs for each class (DROWSY, FOCUSED, UNFOCUSED)
        image_paths = sorted(list(dirs))
        image_paths = image_paths[:size]

        all_paths.extend(image_paths)

    return all_paths


def get_all_image_paths(all_root_paths):
    """
    Function used to get all spectrogram image paths from a list of root directories
    example: './DROWSY/eeg_record/4' gets all images at path 'DROWSY/eeg_record/4/*png'
    :param all_root_paths:
    :return all_image_paths: complete paths to all images we are going to input for train/test data
    """
    all_image_paths = []

    for root_path in all_root_paths:
        png_path = os.path.join(root_path, '*.png')
        image_paths = glob.glob(png_path, recursive=True)

        all_image_paths.extend(image_paths)

    return all_image_paths

def get_train_test_data(images_path_list, image_size):
    """
    Function used to get all train and test images and label data
    :param images_path_list: list of all images we want to read
    :param image_size: image resize to save computation time
    :return data, labels: data with their labels (e.g., DROWSY, FOCUSED, UNFOCUSED)
    """
    data = []
    labels = []

    for image_path in images_path_list:
        image_src = np.asarray(Image.open(image_path).resize((image_size, image_size)).convert('RGB'))
        data.append(image_src)

        label = image_path.split(os.path.sep)[-4]
        labels.append(label)

    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    return data, labels


def main():
    """
    Main Enterance of model
    """

    # handle arguments
    args = handle_arguments()

    print('-------------------------\n[INFO] Preprocessing Data\n-------------------------')

    # Get all paths we want to read data from
    data_paths = determine_data_paths(PATH_TO_DATASET, args.channel, int(args.size))

    all_image_paths = get_all_image_paths(data_paths)

    data_set, labels = get_train_test_data(all_image_paths, int(args.image_size))

    print('-------------------------\n[INFO] Building Model\n-------------------------')

    # split training and test data
    (trainX, testX, trainY, testY) = train_test_split(data_set, labels, test_size=0.25, random_state=42)

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)
    #trainX = lb.fit_transform(trainX)
    #testX = lb.transform(testX)
    print(trainY[:])
    print("from here starts YYY")
    print(testY[:])


    # building model
    model = Sequential()

    # adding layers
    model.add(Conv2D(32, (3, 3), input_shape=(int(args.image_size), int(args.image_size), 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation=ACTIVATION))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.4))
    model.add(Dense(128, activation=ACTIVATION))
    
    #model.add(Dropout(0.4))
    #model.add(Dense(64, activation=ACTIVATION))
    
    #model.add(Dense(250, activation=ACTIVATION))

    # prediction layer, using softmax because we are expecting more than two outcomes (DROWSY, FOCUSED, UNFOCUSED)
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

    print('-------------------------\n[INFO] Train Model\n-------------------------')

    history = model.fit(trainX, trainY, epochs=int(args.epochs), validation_data=(testX, testY), batch_size=128)

    print('-------------------------\n[INFO] Plot Results\n-------------------------')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(testX, testY, verbose=2)

    print('Test Loss: {test_loss}'.format(test_loss=test_loss))
    print('Test Accuracy: {test_acc}'.format(test_acc=test_acc))
    print(test_acc)


if __name__ == '__main__':
    main()
