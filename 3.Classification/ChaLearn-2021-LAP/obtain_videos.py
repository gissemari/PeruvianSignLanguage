# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:51:59 2021

@author: Joe
"""

import glob
import pandas as pd
import shutil
import os
from sys import platform
import argparse

#
parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--allfiles', type=str, default='./../../Data/Videos/Segmented_gestures/', help='...')
args = parser.parse_args()
    
pathAllFiles = args.allfiles
all_files = glob.glob(pathAllFiles + '*/*.mp4')
train_ids = pd.read_csv("./data/train_ids.csv", encoding='utf-8')
val_ids = pd.read_csv("./data/val_ids.csv", encoding='utf-8')
print(train_ids)

for filePath in all_files:

    
    if platform == 'linux' or platform == 'linux2':
        name = filePath.split('/')[-1]
    else:
        name = filePath.split('\\')[-1]
    
    name = name.split('.')[0]


    isVal = False
    for valName, index in val_ids.values.tolist():
        if name == valName:
            isVal = True
            print(name)
            continue

    isTrain = False
    for trainName, index in train_ids.values.tolist():
        if name == trainName:
            isTrain = True
            print(name)
            continue

    if isVal:
        target = './project/data/mp4/val/'+name+'_color.mp4'
        shutil.copyfile(filePath.replace('\\','/'), target)
    if isTrain:
        target = './project/data/mp4/train/'+name+'_color.mp4'
        shutil.copyfile(filePath.replace('\\','/'), target)
