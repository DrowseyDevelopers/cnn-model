import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import glob

PATH_TO_DATASETS = ['./DROWSY/**/4', './FOCUSED/**/4', 'UNFOCUSED/**/4']

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
            image_src = cv2.imread(png)
            image_src = cv2.resize(image_src, (32, 32)).flatten()
            data.append(image_src)
            label = path_to_dataset.split(os.path.sep)[-3]
            labels.append(label)

    print('data length: {0}'.format(len(data)))
    print('labels length: {0}'.format(len(labels)))
if __name__ == '__main__':
    handle_args()

    determine_data_paths(PATH_TO_DATASETS)
