import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog


def extractHogFeatures(imgPath):
    im = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    im1 = cv2.resize(im, (160, 120), interpolation=cv2.INTER_CUBIC)
    hog_features = hog(im1, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return hog_features


mettu_dir = 'D:/College/BTech Project/DataSet/kp/mettu/'

for filename in os.listdir(mettu_dir):
    if os.path.isdir(mettu_dir + filename):
        train_dir = mettu_dir + filename + '/d1/train/'
        X_train = []
        y_train = []
        for cls in os.listdir(train_dir):
            if os.path.isdir(train_dir + cls):
                for img in os.listdir(train_dir + cls):
                    hogFeatures = extractHogFeatures(train_dir + cls + '/' + img)
                    X_train.append(hogFeatures)
                    y_train.append(cls)
        
        test_dir = mettu_dir + filename + '/d1/test/'
        X_test = []
        y_test = []
        for cls in os.listdir(test_dir):
            if os.path.isdir(test_dir + cls):
                for img in os.listdir(test_dir + cls):
                    hogFeatures = extractHogFeatures(test_dir + cls + '/' + img)
                    X_test.append(hogFeatures)
                    y_test.append(cls)

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)

        X_train.to_csv(train_dir+'train.csv', index=False, header=False)
        y_train.to_csv(train_dir+'train_label.csv', index=False, header=False)

        X_test.to_csv(test_dir+'test.csv', index=False, header=False)
        y_test.to_csv(test_dir+'test_label.csv', index=False, header=False)

        print("Mettu " + filename + " d1 done")