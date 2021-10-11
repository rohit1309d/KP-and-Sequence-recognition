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

all_data_dir = 'D:/College/BTech Project/DataSet/kp/all_data/'

train_dir = all_data_dir + 'train/'
X_train = []
y_train = []
for cls in os.listdir(train_dir):
    if os.path.isdir(train_dir + cls):
        for img in os.listdir(train_dir + cls):
            hogFeatures = extractHogFeatures(train_dir + cls + '/' + img)
            print(train_dir + cls + '/' + img + ' completed')
            X_train.append(hogFeatures)
            y_train.append(cls)

print("\nTrain HoG features computed\n")

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

X_train.to_csv(all_data_dir+'X_train.csv', index=False, header=False)
print("\nX_train.csv created\n")
y_train.to_csv(all_data_dir+'y_train.csv', index=False, header=False)
print("\ny_train.csv created\n")

test_dir = all_data_dir + 'test/'
X_test = []
y_test = []
for cls in os.listdir(test_dir):
    if os.path.isdir(test_dir + cls):
        for img in os.listdir(test_dir + cls):
            hogFeatures = extractHogFeatures(test_dir + cls + '/' + img)
            print(test_dir + cls + '/' + img + ' completed')
            X_test.append(hogFeatures)
            y_test.append(cls)

print("\nTest HoG features computed\n")

X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

X_test.to_csv(all_data_dir+'X_test.csv', index=False, header=False)
print("\nX_test.csv created\n")
y_test.to_csv(all_data_dir+'y_test.csv', index=False, header=False)
print("\ny_test.csv created\n")