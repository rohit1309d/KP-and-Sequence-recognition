import os
import numpy as np
import pandas as pd

mettu_dir = 'D:/College/BTech Project/DataSet/kp/mettu/'

X_train_all = []
y_train_all = []
X_test_all = []
y_test_all = []

for filename in os.listdir(mettu_dir):
    if os.path.isdir(mettu_dir + filename):
        X_train = pd.read_csv(mettu_dir + filename + '/d1/train/train.csv', dtype=np.float, header=None).values
        y_train = pd.read_csv(mettu_dir + filename + '/d1/train/train_label.csv', dtype=np.float, header=None).values
        X_test = pd.read_csv(mettu_dir + filename + '/d1/test/test.csv', dtype=np.float, header=None).values
        y_test = pd.read_csv(mettu_dir + filename + '/d1/test/test_label.csv', dtype=np.float, header=None).values

        # X_train_all.append(X_train, ignore_index=True)
        for row in X_train:
            X_train_all.append(row)
        for row in y_train:
            y_train_all.append(row)
        
        for row in X_test:
            X_test_all.append(row)
        for row in y_test:
            y_test_all.append(row)
        
        print("Mettu " + filename + " d1 done")

X_train_all = pd.DataFrame(X_train_all)
X_train_all.to_csv(mettu_dir+'X_train_d1.csv', index=False, header=False)
y_train_all = pd.DataFrame(y_train_all)
y_train_all.to_csv(mettu_dir+'y_train_d1.csv', index=False, header=False)

X_test_all = pd.DataFrame(X_test_all)
X_test_all.to_csv(mettu_dir+'X_test_d1.csv', index=False, header=False)
y_test_all = pd.DataFrame(y_test_all)
y_test_all.to_csv(mettu_dir+'y_test_d1.csv', index=False, header=False)