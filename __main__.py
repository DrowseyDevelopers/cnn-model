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
    classifier = Sequential()


if __name__ == '__main__':
    main()
