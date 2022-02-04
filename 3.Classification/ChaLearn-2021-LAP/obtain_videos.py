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
parser.add_argument('--train', type=int, default=1, help='Create files for train or not, for test')

args = parser.parse_args()
    
pathAllFiles = args.allfiles
all_files = glob.glob(pathAllFiles + '*/*.mp4')

if args.train:
    train_ids = pd.read_csv("./data/train_ids.csv", encoding='utf-8',header=None)
    val_ids = pd.read_csv("./data/val_ids.csv", encoding='utf-8',header=None)
    
    #print(train_ids)

    for filePath in all_files:

        
        if platform == 'linux' or platform == 'linux2':
            name = filePath.split('/')[-1]
        else:
            name = filePath.split('\\')[-1]
        
        name = name.split('.')[0].upper()

        isVal = False
        for newUniqueName, prevName in val_ids.values.tolist():
            if name == newUniqueName:
                isVal = True
                print(name)
                target = './project/data/mp4/val/'+newUniqueName+'_color.mp4'
                shutil.copyfile(filePath.replace('/','\\'), target.replace('/','\\'))
                continue

        isTrain = False
        for newUniqueName, prevName in train_ids.values.tolist():
            if name == newUniqueName:
                isTrain = True
                print(name)
                target = './project/data/mp4/train/'+newUniqueName+'_color.mp4'
                shutil.copyfile(filePath.replace('/', '\\'), target.replace('/','\\'))
                continue
        '''
        if isVal:
            target = './project/data/mp4/val/'+newUniqueName+'_color.mp4'
            shutil.copyfile(filePath.replace('\\','/'), target)
        if isTrain:
            target = './project/data/mp4/train/'+newUniqueName+'_color.mp4'
            shutil.copyfile(filePath.replace('\\','/'), target)
        '''
else:
    test_ids = pd.read_csv("./data/test_ids.csv", encoding='utf-8', header=None)

    #print(all_files)
    for filePath in all_files:
        if platform == 'linux' or platform == 'linux2':
            name = filePath.split('/')[-1]
        else:
            name = filePath.split('\\')[-1]
            
        name = name.split('.')[0].upper()

        isTest = False
        for newUniqueName, prevName in test_ids.values.tolist():
            if name == newUniqueName:
                isTest = True
                print(name)
                target = './project/data/mp4/test/'+newUniqueName+'_color.mp4'
                shutil.copyfile(filePath.replace('\\','/'), target)
                continue

            
            
