import pandas as pd
from lib.timeseries.TimeSeries import TimeSeries
import os

def uv_load(dataset_name,uv_dir):
    try:
        train = {}
        test = {}
          
        train_raw = pd.read_csv((uv_dir + dataset_name + "/" + dataset_name + "_TRAIN.txt"), delim_whitespace=True, header=None)
        test_raw = pd.read_csv((uv_dir + dataset_name + "/" + dataset_name + "_TEST.txt"), delim_whitespace=True, header=None)

        train["Type"] = "UV"
        train["Samples"] = train_raw.shape[0]
        train["Size"] = train_raw.shape[1]-1
        train["Labels"] = []

        test["Type"] = "UV"
        test["Samples"] = test_raw.shape[0]
        test["Size"] = test_raw.shape[1]-1
        test["Labels"] = []

        for i in range(train["Samples"]):
            label = int(train_raw.iloc[i, 0])
            train["Labels"].append(label)
            series = train_raw.iloc[i,1:].tolist()
            train[i] = TimeSeries(series, label)
            train[i].NORM(True)

        for i in range(test["Samples"]):
            label = int(test_raw.iloc[i, 0])
            test["Labels"].append(label)
            series = test_raw.iloc[i, 1:].tolist()
            test[i] = TimeSeries(series, label)
            test[i].NORM(True)


        print("Done reading " + dataset_name + " Training Data...  Samples: " + str(train["Samples"]) + "   Length: " + str(train["Size"]))
        print("Done reading " + dataset_name + " Testing Data...  Samples: " + str(test["Samples"]) + "   Length: " + str(test["Size"]))
        print()

        return train, test

    except:
        print("Data not loaded Try changing the data path in the TimeSeriesLoader file")