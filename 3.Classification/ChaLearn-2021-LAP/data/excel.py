# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 01:53:31 2021

@author: Joe
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection  import train_test_split
import argparse



def getData(path):

    with open(path+'X.data', 'rb') as f:
        X = pickle.load(f)

    with open(path+'X_timeSteps.data', 'rb') as f:
        X_timeSteps = pickle.load(f)

    with open(path+'Y.data', 'rb') as f:
        Y = pickle.load(f)

    with open(path+'weight.data', 'rb') as f:
        weight = pickle.load(f)

    with open(path+'Y_meaning.data', 'rb') as f:
        y_meaning = pickle.load(f)

    return X, Y, weight, y_meaning, X_timeSteps

def getLabelsDataFrame(X_train, y_train, dataDict, y_labels):
    
    name = []
    label = []
    inst_id = []
    
    for idx, dataId in enumerate(X_train):

        for pos, gloss in enumerate(dataDict):
 
            if(dataDict[pos]["gloss"].lower() == y_labels[y_train[idx]].lower()):
                
                for instances in dataDict[pos]["instances"]:

                    if(instances["instance_id"] == dataId):
                        print(instances['timestep_vide_name'])
                        name.append(instances['timestep_vide_name'])
                        label.append(y_train[idx])
                        inst_id.append(dataId)
    
    data = np.stack((name, label), axis=1)
    ids = np.stack((name, inst_id), axis=1)

    dfData = pd.DataFrame(data)
    dfId = pd.DataFrame(ids)

    return dfData, dfId

def main():

    parser = argparse.ArgumentParser(description='Classification')

    parser.add_argument('--dictPath', type=str, default='./../../../Data/Dataset/dict/dict.json', help='...')
    parser.add_argument('--keyPath', type=str, default='./../../../Data/Dataset/readyToRun/', help='...')

    args = parser.parse_args()

    #dictPath = './../../../Data/Dataset/dict/dict.json'
    #keyPath = './../../../Data/Dataset/readyToRun/'
    
    dataDict = pd.read_json(args.dictPath)
    
    x, y, weight, y_labels, x_timeSteps = getData(args.keyPath)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8 , random_state=42, stratify=y)
    
    print("Train size:", len(y_train))
    print("Test size: ", len(y_test))
    
    trainDf, trainId = getLabelsDataFrame(X_train, y_train, dataDict, y_labels)
    print()
    testDf, testId = getLabelsDataFrame(X_test, y_test, dataDict, y_labels)
    
    trainDf.to_csv('train_labels.csv',index=False, header=False)
    testDf.to_csv('val_labels.csv',index=False, header=False)
    
    trainId.to_csv('train_ids.csv',index=False, header=False)
    testId.to_csv('val_ids.csv',index=False, header=False)

main()