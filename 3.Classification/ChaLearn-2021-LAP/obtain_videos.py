# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:51:59 2021

@author: Joe
"""

import glob
import pandas as pd
import shutil
from sys import platform

all_files = glob.glob('./../../Data/Videos/Segmented_gestures/*/*.mp4')

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

    isTrain = True
    for trainName, index in train_ids.values.tolist():
        if name == trainName:
            isTrain = True
            print(name)
            continue

    if isVal:
        target = './project/data/mp4/val/'+name+'_color.mp4'
        shutil.copyfile(filePath, target)
    if isTrain:
        target = './project/data/mp4/train/'+name+'_color.mp4'
        shutil.copyfile(filePath, target)
