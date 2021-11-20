# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:51:59 2021

@author: Joe
"""

import glob
import pandas as pd
import shutil

all_files = glob.glob('./../../Data/Videos/Segmented_gestures/*/*.mp4')

train_ids = pd.read_csv("./data/train_ids.csv", encoding='utf-8')
val_ids = pd.read_csv("./data/val_ids.csv", encoding='utf-8')

for filePath in all_files:
    
    name = filePath.split('\\')[-1]
    name = name.split('.')[0]

    isVal = False
    for valName, index in val_ids.values.tolist():
        if name == valName:
            isVal = True
    print(name)
    if isVal:
        target = './project/data/mp4/val/'+name+'_color.mp4'
        shutil.copyfile(filePath, target)
    else:
        target = './project/data/mp4/train/'+name+'_color.mp4'
        shutil.copyfile(filePath, target)