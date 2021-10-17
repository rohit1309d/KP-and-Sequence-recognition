from joblib import load
import json
from pathlib import Path
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


all_data_dir = 'D:/College/BTech Project/DataSet/kp/all_data/'
annotation_file = 'D:/College/BTech Project/DataSet/AnnotationFiles/'

clf = load('../output/model.joblib')
print("Model loaded")

adavu_seq = pd.read_csv('adavu_seq.csv', dtype=str, header=None).values
mapping = pd.read_csv('./annotation_map.csv', header=None).values

with open('adavu_map.json') as f:
    adavu_map = json.load(f)

print("Map and seq file loaded")

y_orig = []
y_predict = []
edit_dist = []
seq_orig = []
seq_predict = []
X_train = pd.read_csv(all_data_dir + 'X_train.csv', dtype=np.float, header=None)
y_train = pd.read_csv(all_data_dir + 'y_train.csv', dtype=np.float, header=None)
X_test = pd.read_csv(all_data_dir + 'X_test.csv', dtype=np.float, header=None)
y_test = pd.read_csv(all_data_dir + 'y_test.csv', dtype=np.float, header=None)

X_all_data = X_train.append(X_test, ignore_index=True)
y_all_data = y_train.append(y_test, ignore_index=True)

print("Data loaded")

for m in range(1,len(adavu_seq)+1):
    mettu_1 = []

    print(adavu_map[str(m)])
    map_elem = []

    for elem in mapping:
        if elem[3] == adavu_map[str(m)][:-2]:
            map_elem = elem
            break

    try:
        annotation = pd.read_csv(annotation_file + map_elem[1] + '/' + map_elem[1] + '_' + adavu_map[str(m)][-1] + '/' + map_elem[2] + '_' + adavu_map[str(m)][-1] + '_D' + str(1)  + '_S1.csv', header=None, usecols=[0,1,2])
        sub = annotation[2]-annotation[1]
    except Exception as e:
        print(e)
    
    try:
        annotation = pd.read_csv(annotation_file + map_elem[1] + '/' + map_elem[1] + '_' + adavu_map[str(m)][-1] + '/' + map_elem[2] + '_' + adavu_map[str(m)][-1] + '_D' + str(2)  + '_S1.csv', header=None, usecols=[0,1,2])
        sub2 = annotation[2]-annotation[1]
        sub = [x+y+2 for x,y in zip(sub, sub2)]
    except Exception as e:
        print(e)

    try:
        annotation = pd.read_csv(annotation_file + map_elem[1] + '/' + map_elem[1] + '_' + adavu_map[str(m)][-1] + '/' + map_elem[2] + '_' + adavu_map[str(m)][-1] + '_D' + str(3)  + '_S1.csv', header=None, usecols=[0,1,2])
        sub3 = annotation[2]-annotation[1]
        sub = [x+y+1 for x,y in zip(sub, sub3)]
    except Exception as e:
        print(e)

    for i in adavu_seq[m-1]:
        mettu_1.append(y_all_data[0].value_counts().get(float(i)))
    
    print(sub)
    print(mettu_1)
    print(min(min(sub), min(mettu_1)))

    for i in range(min(min(sub), min(mettu_1), 40)):
        n = 80
        map_cls = {}
        mettu_all_data = []
        for j in range(n):
            index = y_all_data.index[y_all_data[0] == np.float(adavu_seq[m-1][j])].tolist()
            if np.float(adavu_seq[m-1][j]) not in map_cls.keys():
                map_cls[np.float(adavu_seq[m-1][j])] = 0
            
            num_img = mettu_1[np.where(adavu_seq[m-1] == adavu_seq[m-1][j])[0][0]]
            map_cls[np.float(adavu_seq[m-1][j])] %= num_img

            mettu_all_data.append(X_all_data.iloc[index].iloc[map_cls[np.float(adavu_seq[m-1][j])]].values)        
            map_cls[np.float(adavu_seq[m-1][j])] += 1
        
        predict_kp_seq = clf.predict(mettu_all_data)
        seq_predict.append(predict_kp_seq)
        y_orig.append(m)
        seq_index, edit_distance = compute_edit_distance(adavu_seq, predict_kp_seq)
        seq_orig.append(adavu_seq[seq_index])
        y_predict.append(seq_index + 1)
        edit_dist.append(edit_distance)
        print(i, m, seq_index+1, edit_distance)

wrong_kp_col = compute_wrong_kps(seq_orig, seq_predict, edit_dist)
Path('../output/all_data').mkdir(parents=True,exist_ok=True)

print("Accuracy - " + str(accuracy_score(y_orig, y_predict)))
conf_matrix = confusion_matrix(y_orig, y_predict)
df = pd.DataFrame(conf_matrix)
df.to_csv('../output/all_data/sequence_confusion_matrix_all_data.csv', index=False, header=False)
print("Confusion matrix saved")

df_orig = pd.DataFrame(seq_orig)
df_orig.to_csv('../output/all_data/sequence_original_all_data.csv', index=False, header=False)
print("sequence_original_all_data saved")

df_predict = pd.DataFrame(seq_predict)
df_predict.to_csv('../output/all_data/sequence_predicted_all_data.csv', index=False, header=False)
print("sequence_predicted_all_data saved")

df_ed = pd.DataFrame(list(zip(y_orig, y_predict, edit_dist, wrong_kp_col)), columns =['Sequence_Original', 'Sequence_Predicted', 'Edit_Distance', 'Mark the wrong KPs that detected in the sequenece during the prediction. Within () the correct KP is given'])
df_ed.to_csv('../output/all_data/sequence_edit_distance_all_data-1.csv',index=False)
print("without names")

y_orig_adavu = list(map(adavu_map.get, list(map(str, y_orig))))
y_predict_adavu = list(map(adavu_map.get, list(map(str, y_predict))))

df_ed = pd.DataFrame(list(zip(y_orig_adavu, y_predict_adavu, edit_dist, wrong_kp_col)), columns =['Sequence_Original', 'Sequence_Predicted', 'Edit_Distance', 'Mark the wrong KPs that detected in the sequenece during the prediction. Within () the correct KP is given'])
df_ed.to_csv('../output/all_data/sequence_edit_distance_all_data.csv',index=False)
print("sequence_edit_distance saved")