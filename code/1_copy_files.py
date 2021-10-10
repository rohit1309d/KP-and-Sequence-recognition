import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

root_dir = 'D:/College/BTech Project/DataSet/'
annotation_file = 'D:/College/BTech Project/DataSet/AnnotationFiles/Kuditta_Mettu'
mettu_data = 'D:/College/BTech Project/DataSet/images/Background_sub_images/mettu/'

Path(root_dir+'kp/mettu').mkdir(parents=True,exist_ok=True)

for filename in os.listdir(mettu_data):
    for folder in os.listdir(mettu_data + filename + '/'):
        for back_img in os.listdir(mettu_data + filename + '/' + folder):
            Path(root_dir + 'kp/mettu/' + filename + '/' + folder + '/train').mkdir(parents=True,exist_ok=True)
            Path(root_dir + 'kp/mettu/' + filename + '/' + folder + '/test').mkdir(parents=True,exist_ok=True)

data_set = 'D:/College/BTech Project/DataSet/kp/mettu/'

for filename in os.listdir(mettu_data):
    
    annotation = pd.DataFrame
    if int(filename) < 4:
        annotation = pd.read_excel(annotation_file + '/Kuditta_Mettu_' + filename + '/mettu_' + filename + '_D1_S1.xlsx', engine='openpyxl', header=None, usecols='A:C', nrows=16)
    else:
        annotation = pd.read_excel(annotation_file + '/Kuditta_Mettu_' + filename + '/mettu_' + filename + '_D1_S1.xlsx', engine='openpyxl', header=None, usecols='A:C', nrows=32)
    
    for index, row in annotation.iterrows():
        kp_class = row[0].split('P')[-1]
        start = int(row[1])
        end = int(row[2])
        mid = start + int((end-start)*0.8)

        origin = mettu_data + filename + '/' + 'd1/back_images/'
        dest = data_set + filename + '/d1/'
        train_dir = dest + '/train/' + kp_class
        test_dir = dest + '/test/' + kp_class
        
        Path(train_dir).mkdir(parents=True,exist_ok=True)
        for i in range(start, mid):
            try:
                shutil.copy(origin + str(i) + '.png', train_dir)
            except:
                print(origin + str(i) + '.png' + ' not found')
        
        Path(test_dir).mkdir(parents=True,exist_ok=True)
        for i in range(mid+1, end):
            try:
                shutil.copy(origin + str(i) + '.png', test_dir)
            except:
                print(origin + str(i) + '.png' + ' not found')
    
