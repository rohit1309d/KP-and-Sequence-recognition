from joblib import load
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def compute_edit_distance(arr, v):
    edit_distances = []
    v1 = np.array(v)
    for a in arr:
        a1 = np.array(a).astype(np.float)
        edit_distances.append(np.sum(a1 != v1))
    edit_distance_index = np.argmin(edit_distances)
    return edit_distance_index, edit_distances[edit_distance_index]

def compute_wrong_kps(seq_orig, seq_predict, edit_dist):
    wrong_kp_col = []
    for i in range(len(seq_orig)):
        val = ""
        if edit_dist[i] == 0:
            val = "NIL,"
        else:
            for j in range(len(seq_orig[i])):
                if float(seq_orig[i][j]) == seq_predict[i][j]:
                    val += seq_orig[i][j] + ","
                else:
                    val += str(int(seq_predict[i][j])) + "*(" + seq_orig[i][j] + "),"
        wrong_kp_col.append(val[:-1])
    return wrong_kp_col

mettu_dir = 'D:/College/BTech Project/DataSet/kp/mettu/'
annotation_file = 'D:/College/BTech Project/DataSet/AnnotationFiles/Kuditta_Mettu'
mettu_data = 'D:/College/BTech Project/DataSet/images/Background_sub_images/mettu/'

clf = load('../output/model.joblib')

mettu = [['94', '3', '95', '3', '96', '97', '95', '3', '98', '99', '95', '3', '96', '97', '95', '3', '94', '3', '95', '3', '96', '97', '95', '3', '98', '99', '95', '3', '96', '97', '95', '3'], 
['101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100', '101', '102', '103', '100'], 
['104', '105', '46', '42', '104', '105', '46', '42', '106', '107', '48', '43', '106', '107', '48', '43', '104', '105', '46', '42', '104', '105', '46', '42', '106', '107', '48', '43', '106', '107', '48', '43'], 
['108', '66', '103', '100', '109', '68', '103', '100', '110', '111', '112', '60', '113', '114', '115', '116', '109', '68', '103', '100', '108', '66', '103', '100', '113', '114', '115', '116', '110', '111', '112', '60']]

y_orig = []
y_predict = []
edit_dist = []
seq_orig = []
seq_predict = []
X_test = pd.read_csv(mettu_dir + 'X_test_d1.csv', dtype=np.float, header=None)
y_test = pd.read_csv(mettu_dir + 'y_test_d1.csv', dtype=np.float, header=None)

for m in range(1,5):
    mettu_1 = []

    for i in mettu[m-1]:
        mettu_1.append(y_test[0].value_counts().get(float(i)))
    
    for i in range(min(mettu_1)):
        n = 16
        if m == 4:
            n = 32
        map_cls = {}
        mettu_test = []
        for j in range(n):
            index = y_test.index[y_test[0] == np.float(mettu[m-1][j])].tolist()
            if np.float(mettu[m-1][j]) not in map_cls.keys():
                map_cls[np.float(mettu[m-1][j])] = 0

            mettu_test.append(X_test.iloc[index].iloc[map_cls[np.float(mettu[m-1][j])]].values)        
            map_cls[np.float(mettu[m-1][j])] += 1
        
        if m != 4:
            mettu_test = mettu_test + mettu_test

        predict_kp_seq = clf.predict(mettu_test)
        seq_predict.append(predict_kp_seq)
        y_orig.append(m)
        seq_index, edit_distance = compute_edit_distance(mettu, predict_kp_seq)
        seq_orig.append(mettu[seq_index])
        y_predict.append(seq_index + 1)
        edit_dist.append(edit_distance)

wrong_kp_col = compute_wrong_kps(seq_orig, seq_predict, edit_dist)
Path('../output/test_data').mkdir(parents=True,exist_ok=True)

print("Accuracy - " + str(accuracy_score(y_orig, y_predict)))
conf_matrix = confusion_matrix(y_orig, y_predict)
df = pd.DataFrame(conf_matrix)
df.to_csv('../output/test_data/sequence_confusion_matrix.csv', index=False, header=False)

df_orig = pd.DataFrame(seq_orig)
df_orig.to_csv('../output/test_data/sequence_original.csv', index=False, header=False)

df_predict = pd.DataFrame(seq_predict)
df_predict.to_csv('../output/test_data/sequence_predicted.csv', index=False, header=False)

df_ed = pd.DataFrame(list(zip(y_orig, y_predict, edit_dist, wrong_kp_col)), columns =['Sequence_Original', 'Sequence_Predicted', 'Edit_Distance', 'Mark the wrong KPs that detected in the sequenece during the prediction. Within () the correct KP is given'])
df_ed.to_csv('../output/test_data/sequence_edit_distance.csv',index=False)
