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
from tensorflow.keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import glob
import random
import os

PATH_TO_DATASET = ['./DROWSY/**', './FOCUSED/**', 'UNFOCUSED/**']


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
    parser.add_argument('-c', '--channel', dest='channel', required=True, choices=[4, 5, 8, 9, 10, 11, 16],
                        help='Flag used to determine what channel we want to create a model for')
    parser.add_argument('-s', '--set', dest='size', required=True,
                        help='Flag used to determine the amount of experiments to import for train/test data')
    parser.add_argument('-i', '--image-size', dest='image_size', required=True,
                        help='Flag used to determine the length and width to resize the data spectrogram images')
    parser.add_argument('-e', '--epoch', dest='epoch', required=True,
                        help='Flag used to determine the number of epochs for training')
    args = parser.parse_args()

    return args


def get_dataset(path_to_datasets):
    """
    Function used to get training data and test data
    :param path_to_datasets: list of paths to data
    :return:
    """

    for path_to_dataset in path_to_datasets:
        dirs = glob.glob(path_to_datasets, recursive=True)

        image_paths = sorted(list(dirs))
        random.seed(42)
        random.shuffle(image_paths)


def main():
    """
    Main Enterance of model
    """
    handle_arguments()

if __name__ == '__main__':
    main()
