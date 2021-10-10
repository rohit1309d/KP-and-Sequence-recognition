from joblib import load
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def miss_predicted_class(y_true, conf_matrix):
    labels = np.unique(y_true)
    labels_dict = dict(zip(labels,conf_matrix))
    miss_predicted_col = []
    for key in labels_dict.keys():
        class_dict = dict(zip(labels,labels_dict[key]))
        m = ''
        for k in class_dict.keys():
            if key != k and class_dict[k] != 0:
                    m = m + (str(int(k)) + '(' + str(class_dict[k]) + ')  ')
            
        miss_predicted_col.append(m)
    return miss_predicted_col
                   

all_data_dir = 'D:/College/BTech Project/DataSet/kp/all_data/'
clf = load('../output/model.joblib')

X_test = pd.read_csv(all_data_dir + 'X_test.csv', dtype=np.float, header=None).values
y_test = pd.read_csv(all_data_dir + 'y_test.csv', dtype=np.float, header=None).values
y_test = list(itertools.chain(*y_test))

y_predict = clf.predict(X_test)
print("Accuracy - " + str(accuracy_score(y_test, y_predict)))

conf_matrix = confusion_matrix(y_test, y_predict)
df = pd.DataFrame(conf_matrix)
df.to_csv('../output/kp_confusion_matrix.csv', index=False, header=False)

y_train = pd.read_csv(all_data_dir + 'y_train.csv', dtype=np.float, header=None).values

train_cols = np.unique(y_train, return_counts=True)
train_col = pd.DataFrame(list(zip(train_cols[0], train_cols[1])), columns=['class', 'count_sample'])

test_cols = np.unique(y_test, return_counts=True)
test_col = pd.DataFrame(list(zip(test_cols[0], test_cols[1])), columns=['class', 'count_sample'])

count_tt = train_col.merge(test_col, on='class', suffixes=('_train', '_test'))

mis_predict = conf_matrix.sum(axis=1) - conf_matrix.diagonal()
acc_col = (conf_matrix.diagonal()*100)/conf_matrix.sum(axis=1)

count_tt = count_tt.assign(count_miss_predicted = mis_predict, accuracy = acc_col)

miss_predict_col = miss_predicted_class(y_train, conf_matrix)
final_result = count_tt.assign(miss_predicted_with = miss_predict_col)

final_result.to_csv('../output/kp_test_analysis.csv', index=False)