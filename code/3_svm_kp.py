import time
import itertools
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump

all_data_dir = 'D:/College/BTech Project/DataSet/kp/all_data/'

begin = time.time()

X_train = pd.read_csv(all_data_dir + 'X_train.csv', dtype=np.float, header=None).values
print("X_train reading done")
y_train = pd.read_csv(all_data_dir + 'y_train.csv', dtype=np.float, header=None).values
print("y_train reading done")
X_test = pd.read_csv(all_data_dir + 'X_test.csv', dtype=np.float, header=None).values
print("X_test reading done")
y_test = pd.read_csv(all_data_dir + 'y_test.csv', dtype=np.float, header=None).values
print("y_test reading done")

y_train = list(itertools.chain(*y_train))
y_test = list(itertools.chain(*y_test))

clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto', verbose=1))
clf.fit(X_train, y_train)

dump(clf, '../output/model.joblib')

end = time.time()
print(f"Total runtime of the program is {end - begin}")