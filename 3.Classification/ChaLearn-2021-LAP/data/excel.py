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
from collections import Counter


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

    prevNameID = []
    label = []
    newNameUniqueID = []
    nFrames = []

    # for a key in X_train
    for idx, dataId in enumerate(X_train):

        # find in dataDict his respective gloss
        for pos, gloss in enumerate(dataDict):

            if(dataDict[pos]["gloss"].upper() == y_labels[y_train[idx]].upper()):

                for instances in dataDict[pos]["instances"]:

                    if(instances["instance_id"] == dataId):
                        #print(instances['timestep_vide_name'])
                        print(instances['unique_name'])
                        newNameUniqueID.append(instances['unique_name'].upper())
                        prevNameID.append(instances['timestep_vide_name'].upper())
                        nFrames.append(instances['frame_end'])
                        label.append(y_train[idx])
                        #inst_id.append(dataId)

    # Used to get videos from mp4, and other ids. The only prev not unique ID is in file ids.csv
    nVidFrames = np.stack((newNameUniqueID, nFrames), axis=1)
    data = np.stack((newNameUniqueID, label), axis=1)
    ids = np.stack((newNameUniqueID, prevNameID), axis=1)
    dfData = pd.DataFrame(data)
    dfId = pd.DataFrame(ids)
    dfNFrames = pd.DataFrame(nVidFrames)
    return dfData, dfId, dfNFrames

def main():

    parser = argparse.ArgumentParser(description='Classification')

    parser.add_argument('--dictPath', type=str, default='./../../../Data/Dataset/dict/dict.json', help='...')
    parser.add_argument('--keyPath', type=str, default='./../../../Data/Dataset/readyToRun/', help='...')
    parser.add_argument('--splitRate', type=float, default=0.8, help='Percentage for training')

    args = parser.parse_args()

    #dictPath = './../../../Data/Dataset/dict/dict.json'
    #keyPath = './../../../Data/Dataset/readyToRun/'
    
    dataDict = pd.read_json(args.dictPath)
    
    x, y, weight, y_labels, x_timeSteps = getData(args.keyPath)
    
    #if creating split for train set, and val
    
    print("Split rate ", args.splitRate)
    if args.splitRate<1.0:
    
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=args.splitRate , random_state=42, stratify=y)
        
        print("Train size:", len(y_train))
        print("Test size: ", len(y_test))
        
        trainDf, trainId, trainNFrames = getLabelsDataFrame(X_train, y_train, dataDict, y_labels)
        print()
        testDf, testId, testNFrames = getLabelsDataFrame(X_test, y_test, dataDict, y_labels)
        
        trainDf.to_csv('train_labels.csv',index=False, header=False)
        testDf.to_csv('val_labels.csv',index=False, header=False)

        trainNFrames.to_csv('train_nframes.csv',index=False, header=False)
        testNFrames.to_csv('val_nframes.csv',index=False, header=False)

        trainId.to_csv('train_ids.csv',index=False, header=False)
        testId.to_csv('val_ids.csv',index=False, header=False)
    # if all the data is for training, or testing for NUEVO PUCP
    else:
        testDf, testId, testNFrames = getLabelsDataFrame(x, y, dataDict, y_labels)
        testDf.to_csv('test_labels.csv',index=False, header=False)
        testNFrames.to_csv('test_nframes.csv',index=False, header=False)
        testId.to_csv('test_ids.csv',index=False, header=False)

main()
