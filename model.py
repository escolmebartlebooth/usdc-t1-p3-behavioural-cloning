""" behvioural cloning model """

# imports
import cv2
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Activation, Dense, Flatten
from keras.layers import Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
# from keras.utils.visualize_util import plot
import numpy as np

# global file locations
FILE_DIR = "usdc-t1-p3-data/data/"
# FILE_DIR = "usdc-t1-p3-data/"
DATA_FILE = "driving_log.csv"
CORRECTED_PATH = FILE_DIR + "IMG/"
# MODEL_TO_USE = nvidia_2

# to arrive at correct data size after image augmentation
SAMPLES_FACTOR = 3

# number of epochs
NB_EPOCHS = 2

# update this value if path info is windows (w) \ or linux (l) /
FILE_FROM = "l"
# FILE_FROM = "w"


def read_data_from_file():
    """ function to read in image data from csv
        and to correct for image folder
        returns a list of images for use in training
    """

    data_list = []
    with open(FILE_DIR+DATA_FILE, 'rt') as f:
        # ignore first line if header
        img_data = csv.reader(f)
        firstline = 0
        for line in img_data:
            if firstline == 0:
                firstline = 1
            else:
                data_list.append(line)

    print(len(data_list))

    train_data, validation_data = train_test_split(data_list, test_size=0.2)
    return train_data, validation_data


def generate_data(X, file_from="l", batch_size=32):
    """
        generator function for training and validation data
    """

    sample_size = len(X)

    # run forever...
    while 1:
        # shuffle the data
        shuffle(X)
        # generate a sample batch
        for offset in range(0, sample_size, batch_size):
            # slice off the next batch
            batch_samples = X[offset:offset+batch_size]

            # placeholders for the images and angles
            features = []
            measurements = []

            # loop the batch
            for item in batch_samples:
                # add centre, left and right images and adjust steering
                for i in range(3):
                    # check whether data from windows or linux
                    if file_from == "w":
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[i].split("\\")[-1]))
                    else:
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[i].split("/")[-1]))
                    if i == 0:
                        correction_factor = 0
                    elif i == 1:
                        correction_factor = 0.25
                    else:
                        correction_factor = -0.25
                    angle = float(item[3])
                    measurements.append(angle+correction_factor)

                # now build augmented images
                aug_features, aug_measurements = [], []
                for feature, measurement in zip(features, measurements):
                    aug_features.append(feature)
                    aug_measurements.append(measurement)
                    # now also add a flipped image
                    aug_features.append(cv2.flip(feature, 1))
                    aug_measurements.append(measurement*-1.0)

                yield shuffle(np.array(aug_features),
                              np.array(aug_measurements))

def generate_data_2(X, file_from="l", batch_size=32):
    """
        generator function for training and validation data
        this adds more non zero angle images
    """

    sample_size = len(X)

    # run forever...
    while 1:
        # shuffle the data
        shuffle(X)
        # generate a sample batch
        for offset in range(0, sample_size, batch_size):
            # slice off the next batch
            batch_samples = X[offset:offset+batch_size]

            # placeholders for the images and angles
            features = []
            measurements = []

            # loop the batch
            for item in batch_samples:
                # add centre image for zero angle
                angle = float(item[3])
                ch = random.choice([0, 1, 2])
                if angle == 0:
                    if file_from == "w":
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[ch].split("\\")[-1]))
                    else:
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[ch].split("/")[-1]))
                    if ch = 0: measurements.append(angle)
                    if ch = 1: measurements.append(angle+0.25)
                    if ch = 2: measurements.append(angle-0.25)
                else:
                    # add and augment non-zero
                    # left Image
                    if file_from == "w":
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[1].split("\\")[-1]))
                    else:
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[1].split("/")[-1]))
                    measurements.append(angle+0.25)
                    # right image
                    if file_from == "w":
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[2].split("\\")[-1]))
                    else:
                        features.append(cv2.imread(CORRECTED_PATH +
                                                   item[2].split("/")[-1]))
                    measurements.append(angle-0.25)

                # now build augmented images
                aug_features, aug_measurements = [], []
                for feature, measurement in zip(features, measurements):
                    aug_features.append(feature)
                    aug_measurements.append(measurement)
                    # now also add a flipped image for non zero angles
                    if measurement != 0:
                        aug_features.append(cv2.flip(feature, 1))
                        aug_measurements.append(measurement*-1.0)

                yield shuffle(np.array(aug_features),
                              np.array(aug_measurements))


def training_model(X_train, X_valid):
    """
        function to train model
        args: training and validation data files
    """
    # create data generators
    batch_size = 32
    X_gen_train = generate_data(X_train, FILE_FROM, batch_size=batch_size)
    X_gen_valid = generate_data(X_valid, FILE_FROM, batch_size=batch_size)

    # create model
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
              input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    # plot(model, to_file='examples/model.png')
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(X_gen_train,
                        samples_per_epoch=len(X_train) * SAMPLES_FACTOR,
                        nb_epoch=NB_EPOCHS, validation_data=X_gen_valid,
                        nb_val_samples=len(X_valid) * SAMPLES_FACTOR)
    model.save("model.h5")
    for item in history.history.keys():
        print("key val: {0} is {1}".format(item,
                                           history.history[item]))

def training_model_2(X_train, X_valid):
    """
        function to train model
        args: training and validation data files
    """
    # create data generators
    batch_size = 32
    X_gen_train = generate_data_2(X_train, FILE_FROM, batch_size=batch_size)
    X_gen_valid = generate_data_2(X_valid, FILE_FROM, batch_size=batch_size)

    # create model
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5,
              input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.5))
    model.add(Dense(50))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    # plot(model, to_file='examples/model.png')
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(X_gen_train,
                        samples_per_epoch=len(X_train) * SAMPLES_FACTOR,
                        nb_epoch=NB_EPOCHS, validation_data=X_gen_valid,
                        nb_val_samples=len(X_valid) * SAMPLES_FACTOR)
    model.save("model.h5")
    for item in history.history.keys():
        print("key val: {0} is {1}".format(item,
                                           history.history[item]))

if __name__ == "__main__":
    train_data, validation_data = read_data_from_file()
    training_model_2(train_data, validation_data)
