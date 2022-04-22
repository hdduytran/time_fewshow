import sys
import os
import sklearn
from sklearn import preprocessing
sys.path.append(os.getcwd()[:-5])
from lib.timeseries.TimeSeriesLoader import uv_load
import pickle
from scipy.sparse import csr_matrix
import numpy as np
from pyts.transformation import BOSS
from os.path import dirname
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def feature_data_writer(dataset_name, train_ratio, ind ,X_train_feature, y_train, X_test_feature, y_test):
    father_path = './feature_data/' + dataset_name+ '/'+str(train_ratio)+'/'+str(ind)+'/'
    if not os.path.exists(father_path):
        os.makedirs(father_path)
        

    dictionary = {'X_train_feature': X_train_feature,
                  'y_train': y_train,
                  'X_test_feature': X_test_feature,
                  'y_test': y_test}
    
   
    save_path = father_path+ dataset_name + '.npy'
    np.save(save_path, dictionary)


def TSC_data_loader(dataset_name):
    uv_dir = "./data/"
    Train_dataset = np.loadtxt(uv_dir + dataset_name + '/' + dataset_name + '_TRAIN.txt')
    Test_dataset = np.loadtxt(uv_dir + dataset_name + '/' + dataset_name + '_TEST.txt')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)
    

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    
    #X_train = np.nan_to_num(X_train) 
    #X_test  = np.nan_to_num(X_test)

    return X_train, y_train, X_test, y_test

name_list=[
  'ArrowHead',
#   # 'ECG200',
#   'BME',
#   'CBF',
#   'Chinatown',
#   'GunPoint',
]


# train_ratio_list = [0.1]
train_ratio_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


  
for name in  name_list:
    X_train, y_train, X_test, y_test = TSC_data_loader(name)
    feature_data_writer(name,1,10, X_train, y_train, X_test, y_test)    
    
    for train_ratio in train_ratio_list:    
        X_train_ori, y_train_ori, X_test, y_test = TSC_data_loader(name)
        sss = StratifiedShuffleSplit(n_splits=10, test_size=1 - train_ratio, random_state=0)
        sss.get_n_splits(X_train_ori, y_train_ori)
        ind = 0
        for train_index, test_index in sss.split(X_train_ori, y_train_ori):
            print(ind)
            X_train = X_train_ori[train_index,:]
            y_train = y_train_ori[train_index]
            
            print('X_train shape: ',X_train.shape)
            
            feature_data_writer(name,train_ratio,ind, X_train, y_train, X_train, y_test)
            ind = ind+1