import itertools
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

mettu_dir = 'D:/College/BTech Project/DataSet/kp/mettu/'

X_train = pd.read_csv(mettu_dir + 'X_train_d1.csv', dtype=np.float, header=None).values
y_train = pd.read_csv(mettu_dir + 'y_train_d1.csv', dtype=np.float, header=None).values
X_test = pd.read_csv(mettu_dir + 'X_test_d1.csv', dtype=np.float, header=None).values
y_test = pd.read_csv(mettu_dir + 'y_test_d1.csv', dtype=np.float, header=None).values

y_train = list(itertools.chain(*y_train))
y_test = list(itertools.chain(*y_test))

clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
clf.fit(X_train, y_train)

dump(clf, '../output/model.joblib') 
