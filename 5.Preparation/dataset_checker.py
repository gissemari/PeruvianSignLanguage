#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:35:25 2021

@author: joe
"""
# Standard library imports
import argparse

# Third party imports
import numpy as np
from sklearn.model_selection  import train_test_split
import pandas as pd

# Local imports
from utils import LoadData

parser = argparse.ArgumentParser(description='Classification')

parser.add_argument('--keys_input_Path', type=str,
                    default="./Data/Dataset/readyToRun/",
                    help='relative path of key input.'
                    ' Default: ./Data/Dataset/readyToRun/')

parser.add_argument('--dict_Path', type=str,
                    default="./Data/Dataset/dict/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/dict/dict.json')

args = parser.parse_args()

dataDict = pd.read_json(args.dict_Path)

x, y, weight, y_labels, x_timeSteps = LoadData.getData(args.keys_input_Path)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8 , random_state=42, stratify=y)

print("Train size:", len(y_train))
print("Test size: ", len(y_test))

count = 0

for idx, dataId in enumerate(X_train):
    print(dataId)
    indG = -1
    for pos, gloss in enumerate(dataDict):

        if(dataDict[pos]["gloss"] == y_labels[y_train[idx]].lower().lower()):
            indG = pos
    
    toCheck = [instances["instance_id"] for instances in dataDict[indG]["instances"]]
    
    if not dataId in toCheck:
        count +=1

for idx, dataId in enumerate(X_test):
    
    indG = -1
    for pos, gloss in enumerate(dataDict):

        if(dataDict[pos]["gloss"] == y_labels[y_test[idx]].lower()):
            indG = pos

    toCheck = [instances["instance_id"] for instances in dataDict[indG]["instances"]]

    if not dataId in toCheck:
        count +=1

print("There are %d errors in the dataset" % count)



