""" behvioural cloning model """

# imports
import cv2
import csv
from keras.layers import Dense, Flatten
from keras.models import Sequential
import numpy as np


def read_data_from_file():
    """ function to read in image data from csv
        and to correct for image folder
        returns a list of images for use in training
    """

    FILE_DIR = "usdc-t1-p3-data/"
    DATA_FILE = "driving_log.csv"
    CORRECTED_PATH = "usdc-t1-p3-data/IMG/"
    data_list = []
    with open(FILE_DIR+DATA_FILE,'rt') as f:
        img_data = csv.reader(f)
        for line in img_data:
            data_list.append(line)

    # now extract images and measurements
    features = []
    measurements = []

    for item in data_list:
        features.append(cv2.imread(CORRECTED_PATH+item[0].split("\\")[-1]))
        measurements.append(item[3])

    return np.array(features), np.array(measurements)

def training_model(X,y):
    """ function to train model """
    model = Sequential()
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X,y,validation_split=0.2, shuffle=True, nb_epoch=3)
    model.save("model.h5")


if __name__ == "__main__":
    features, measurements = read_data_from_file()
    training_model(features, measurements)
