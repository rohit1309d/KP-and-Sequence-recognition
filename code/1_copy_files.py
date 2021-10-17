import os
from pathlib import Path
import shutil
import pandas as pd

root_dir = 'D:/College/BTech Project/DataSet/'
annotation_file = 'D:/College/BTech Project/DataSet/AnnotationFiles/'
images_dir = 'D:/College/BTech Project/DataSet/images/Background_sub_images/'
data_set = 'D:/College/BTech Project/DataSet/kp/all_data/'

mapping = pd.read_csv('./annotation_map.csv', header=None).values

Path(root_dir+'kp/all_data').mkdir(parents=True,exist_ok=True)
Path('../output').mkdir(parents=True,exist_ok=True)
Path(root_dir + 'kp/all_data/train').mkdir(parents=True,exist_ok=True)
Path(root_dir + 'kp/all_data/test').mkdir(parents=True,exist_ok=True)

for map_elem in mapping:
    adavu_img = images_dir + map_elem[0] + "/"
    for filename in os.listdir(adavu_img):
        for di in range(3):
            annotation = pd.DataFrame
            annotation_exists = 1
            try:
                annotation = pd.read_csv(annotation_file + map_elem[1] + '/' + map_elem[1] + '_' + filename + '/' + map_elem[2] + '_' + filename + '_D' + str(di+1)  + '_S1.csv', header=None, usecols=[0,1,2])
            except Exception as e:
                annotation_exists = 0
                print(e)

            if annotation_exists == 1:
                for index, row in annotation.iterrows():
                    kp_class = row[0].split('P')[-1]
                    if 'B' in kp_class:
                        kp_class = kp_class.split('B')[0]
                    kp_class = str(int(kp_class))
                    start = int(row[1])
                    end = int(row[2])
                    mid = start + int((end-start)*0.8)

                    origin = adavu_img + filename + '/' + 'd' + str(di+1)  + '/back_images/'
                    train_dir = data_set + '/train/' + kp_class
                    test_dir = data_set + '/test/' + kp_class
                    
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
                
