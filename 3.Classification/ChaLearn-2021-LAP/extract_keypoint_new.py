# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:05:09 2021

@author: Joe
"""
import pandas as pd
import pickle
import numpy as np
import os
import glob
import json
import argparse


def openPoseDict(xP, yP, xLH, yLH, xRH, yRH, xF, yF):
    
    zP = [0.0 for _ in range(len(xP))]
    zLH = [0.0 for _ in range(len(xLH))]
    zRH = [0.0 for _ in range(len(xRH))]
    zF = [0.0 for _ in range(len(xF))]

    opd = {"version":1.2,
     "people":[
         {"pose_keypoints_2d":[item for sublist in zip(xP, yP, zP) for item in sublist][:-1],
          "face_keypoints_2d":[item for sublist in zip(xF, yF, zF) for item in sublist][:-1],
          "hand_left_keypoints_2d":[item for sublist in zip(xLH, yLH, zLH) for item in sublist][:-1],
          "hand_right_keypoints_2d":[item for sublist in zip(xRH, yRH, zRH) for item in sublist][:-1],
          "pose_keypoints_3d":[],
          "face_keypoints_3d":[],
          "hand_left_keypoints_3d":[],
          "hand_right_keypoints_3d":[]
          }
         ]
     }
   
    return opd


def extratXYFromBodyPart(fileData, bodyName, exclusivePoints=[]):

    if exclusivePoints: #256.0
        x = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["x"]) if pos in exclusivePoints]
        y = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["y"]) if pos in exclusivePoints]
    else:

        x = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["x"])]
        y = [item * 220.0 for pos, item in enumerate(fileData[bodyName]["y"])]

    return x, y

def keypointsFormat(fileData):

    temp = np.moveaxis(fileData,0,1)
    x = temp[0].astype(float)
    y = temp[1].astype(float)

    xP, yP, xLH, yLH, xRH, yRH, xF, yF = x[0:7], y[0:7], x[7:18], y[7:18], x[18:29], y[18:29], [], []
    
    opd = openPoseDict(xP, yP, xLH, yLH, xRH, yRH, xF, yF)
    #print(opd)
    return opd

def saveJson(ids,output_dir, data, dataType):


    for uniqueName, prevName in ids.values.tolist():

        os.makedirs('%s%s/%s_color.kp/'%(output_dir,dataType,uniqueName), exist_ok=True)

        unique = data.loc[data['names']==uniqueName]['data']


        for instance in unique:

            for pos, timestep in enumerate(instance):

                opd = keypointsFormat(timestep)
                jsonName = '%s%s/%s_color.kp/%s_color_%1.12d_keypoints.json'%(output_dir,dataType,uniqueName,uniqueName,pos)
                
                with open(jsonName, 'w') as f:
                    json.dump(opd, f)


def main():

    parser = argparse.ArgumentParser(description='Classification')

    parser.add_argument('--src', type=str, default='./../.././Data/Dataset/keypoints/', help='...')
    parser.add_argument('--keyPath', type=str, default='./../../../Data/Dataset/readyToRun/', help='...')
    parser.add_argument('--train', type=int, default=1, help='Create files for train or not, for test')

    #src = './../.././Data/Dataset/keypoints/'
    #keyPath = './../../../Data/Dataset/readyToRun/'
    args = parser.parse_args()
    
    output_dir = './project/data/kp/'
    
    if args.train==1:

        train_data = pd.read_pickle("../../Data/merged/AEC-PUCP_PSL_DGI156/merge-train.pk")
        val_data = pd.read_pickle("../../Data/merged/AEC-PUCP_PSL_DGI156/merge-val.pk")

        train_ids = pd.read_csv("./data/train_ids.csv", encoding='utf-8',header=None)
        val_ids = pd.read_csv("./data/val_ids.csv", encoding='utf-8',header=None)

        saveJson(train_ids, output_dir, train_data, "train")
        saveJson(val_ids, output_dir, val_data, "val")
    else:
        test_ids = pd.read_csv("./data/test_ids.csv", encoding='utf-8',header=None)
        saveJson(test_ids, output_dir, args.src, "test")

main()
