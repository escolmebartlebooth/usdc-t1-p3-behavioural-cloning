""" behvioural cloning model """

# imports
import cv2
import csv
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Lambda
from keras.models import Sequential
import numpy as np

# global file locations
FILE_DIR = "usdc-t1-p3-data/data/"

def read_data_from_file():
    """ function to read in image data from csv
        and to correct for image folder
        returns a list of images for use in training
    """

    DATA_FILE = "driving_log.csv"

    data_list = []
    with open(FILE_DIR+DATA_FILE,'rt') as f:
        # ignore first line if header
        img_data = csv.reader(f)
        firstline = 0
        for line in img_data:
            if firstline == 0:
               firstline = 1
            else:
               data_list.append(line)

    train_data, validation_data = train_test_split(data_list, test_size=0.2)
    return train_data, validation_data

def transform_data(X):
    features = []
    measurements = []

    CORRECTED_PATH= FILE_DIR+"IMG/"
    for item in X:
        features.append(cv2.imread(CORRECTED_PATH+item[0].split("/")[-1]))
        measurements.append(item[3])

    print(features[0])
    return np.array(features), np.array(measurements)

def training_model(X,y):
    """ function to train model """
    print(features.shape, measurements.shape)
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160,320,3),
        output_shape=(160,320,3)))
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X,y,validation_split=0.2, shuffle=True, nb_epoch=3)
    model.save("model.h5")


if __name__ == "__main__":
    train_data, validation_data = read_data_from_file()
    features, measurements = transform_data(train_data)
    training_model(features, measurements)
