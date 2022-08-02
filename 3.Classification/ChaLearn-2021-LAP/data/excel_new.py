# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 01:53:31 2021

@author: Joe
"""
import pandas as pd
import numpy as np
import argparse


def getLabelsDataFrame(X_train, y_train, names, y_labels):

    prevNameID = []
    label = []
    newNameUniqueID = []
    nFrames = []

    # for a key in X_train
    for idx, kp in enumerate(X_train):

        newNameUniqueID.append(names[idx])
        prevNameID.append(names[idx])
        nFrames.append(len(kp))
        label.append(y_train[idx])


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

    parser.add_argument('--dictPath', type=str, default='./../../../Data/AEC/dict.json', help='...')
    parser.add_argument('--keyPath', type=str, default='./../../../Data/Dataset/readyToRun/', help='...')
    parser.add_argument('--splitRate', type=float, default=0.8, help='Percentage for training')

    args = parser.parse_args()

    #dictPath = './../../../Data/Dataset/dict/dict.json'
    #keyPath = './../../../Data/Dataset/readyToRun/'
    
    dataDict = pd.read_json(args.dictPath)

    y_labels = pd.read_json('../../../Data/merged/AEC-PUCP_PSL_DGI156/meaning.json')
    y_labels = {v:k for k, v in y_labels[0].items()}

    names_train = pd.read_json("../../../Data/merged/AEC-PUCP_PSL_DGI156/names-train.json")[0]
    names_test = pd.read_json("../../../Data/merged/AEC-PUCP_PSL_DGI156/names-val.json")[0]
    #x, y, weight, y_labels, x_timeSteps = getData(args.keyPath)

    #if creating split for train set, and val
    
    print("Split rate ", args.splitRate)
    
    if args.splitRate<1.0:

        train_data = pd.read_pickle("../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-train.pk")
        val_data = pd.read_pickle("../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-val.pk")

        X_train = train_data['data']
        y_train = train_data['labels']

        X_test = val_data['data']
        y_test = val_data['labels']

        print("Train size:", len(y_train))
        print("Test size: ", len(y_test))

        trainDf, trainId, trainNFrames = getLabelsDataFrame(X_train, y_train, names_train, y_labels)
        print()
        testDf, testId, testNFrames = getLabelsDataFrame(X_test, y_test, names_test, y_labels)
        
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
