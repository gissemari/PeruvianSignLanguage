#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 01:20:51 2021

@author: joe
"""
# Standard library imports
import argparse
import os
from collections import Counter
from shutil import copy2
import json

# Third party imports
import pandas as pd
import numpy as np

# Local imports
import utils.video as vid

parser = argparse.ArgumentParser(description='Dataset converter')

# Path to folder with videos
parser.add_argument('--main_folder_Path', type=str,
                    default="./Data/Keypoints/pkl_TGCN/Segmented_gestures/")

parser.add_argument('--main_video_path', type=str,
                    default="./Data/Videos/Segmented_gestures/")

# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./threeParty-converter/WLASL/data/")

# Number of top words
parser.add_argument("--words", type=int, default=10,
                    help="Number of top words")

args = parser.parse_args()

# return a list of 2D tuple
foldersToLoad = os.listdir(args.main_folder_Path)

wordList = []
timeStepDict = {}

for folderName in foldersToLoad:

    # if the file have extension then that file will be omited
    if(os.path.splitext(folderName)[1] != ''):
        continue

    folder = os.listdir(args.main_folder_Path+folderName)

    # wordList => list that acumulate the name of all files of all folders.
    # The acumulated names will have some modification made by
    # file.split('_')[0]
    #
    # file.split('_')[0] => get the name of the file without "_" and
    # its number. then, these names are acumulated in a list that will
    # be added in wordList
    #
    # timeStepDict => dictionary that acumulate the number of timestep  of
    # each file in its respective word

    for file in folder:

        word = file.split('_')[0]

        fileData = pd.read_pickle(args.main_folder_Path+folderName+'/'+file)

        if word in timeStepDict:
            timeStepDict[word] = timeStepDict[word] + [len(fileData)]
            wordList = wordList + [word]
        else:
            timeStepDict.update({word: [len(fileData)]})

# Convert the given list into dictionary
# the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
wordTrends = Counter(wordList)

# most_common function create a 2D tuple of the N most common words of
# the Counter

topWords = wordTrends.most_common(args.words)
print('Total of considered words: ',len(topWords))
print(topWords)
# to acumulate timesteps to do statistic
union = []

for word, value in topWords:
    union = union + timeStepDict[word]

topWordList = [key for (key, value) in topWords]

wordLabels = list(range(len(topWordList)))

# To identify which word corresponds to its label (int: 1, 2, 3, ...)
#
# topWordList will be the key list
# wordLabels will be the values list
topWordDict = dict(zip(topWordList, wordLabels))


bigAsl = []
vid.createFolder(args.output_Path)
vid.createFolder(args.output_Path+'splits')
vid.createFolder(args.output_Path+'WLASL_VID')
vid.createFolder(args.output_Path+'pose_per_individual_videos')


for folderName in foldersToLoad:

    # if the file have extension then that file will be omited
    if(os.path.splitext(folderName)[1] != ''):
        continue

    folder = os.listdir(args.main_folder_Path+folderName)

    # wordList => list that acumulate the name of all files of all folders.
    # The acumulated names will have some modification made by
    # file.split('_')[0]
    #
    # file.split('_')[0] => get the name of the file without "_" and
    # its number. then, these names are acumulated in a list that will
    # be added in wordList
    #
    count = 0
    for file in folder:
        print("Processing: ", file)
        videoName = file.split('.')[0]

        word = file.split('_')[0]

        # To process only topWordList
        if word not in topWordList:
            continue

        fileData = pd.read_pickle(args.main_folder_Path+folderName+'/'+file)
        vid.createFolder(args.output_Path+'pose_per_individual_videos/'+videoName)
       

        ###################################
        # Splits
        glossPos = -1

        for indG, gloss in enumerate(bigAsl):
            if(gloss["gloss"] == word):
                glossPos = indG

        if glossPos > -1:
            
            # to separate Train and Validation
            wordInstMaxSize = topWords[-1][1]

            wordTrain = int(wordInstMaxSize*0.75)
            wordVal = int(wordInstMaxSize*0.95)
            
            if(len(bigAsl[glossPos]["instances"]) < wordTrain):
                glossInst = {
                    "bbox": [
                        137,
                        16,
                        492,
                        480
                    ],
                    "frame_end": len(fileData),
                    "frame_start": 1,
                    "instance_id": count,
                    "signer_id": 0,
                    "source": "LSP",
                    "split": "train",
                    "url": "",
                    "variation_id": 0,
                    "video_id": videoName
                }
            elif(len(bigAsl[glossPos]["instances"]) < wordVal):        
                glossInst = {
                    "bbox": [
                        137,
                        16,
                        492,
                        480
                    ],
                    "frame_end": len(fileData),
                    "frame_start": 1,
                    "instance_id": count,
                    "signer_id": 0,
                    "source": "LSP",
                    "split": "val",
                    "url": "",
                    "variation_id": 0,
                    "video_id": videoName
                }
            else:
                glossInst = {
                    "bbox": [
                        137,
                        16,
                        492,
                        480
                    ],
                    "frame_end": len(fileData),
                    "frame_start": 1,
                    "instance_id": count,
                    "signer_id": 0,
                    "source": "LSP",
                    "split": "test",
                    "url": "",
                    "variation_id": 0,
                    "video_id": videoName
                }
                
            
            bigAsl[glossPos]["instances"].append(glossInst)

        else:
            glossDict = {"gloss":str(word),
                         "instances":[{
                             "bbox": [
                                 137,
                                 16,
                                 492,
                                 480],
                            "frame_end": len(fileData),
                            "frame_start": 1,
                            "instance_id": count,
                            "signer_id": 0,
                            "source": "LSP",
                            "split": "train",
                            "url": "",
                            "variation_id": 0,
                            "video_id": videoName}]
                         }

            bigAsl.append(glossDict)
        count = count + 1
        
        ###################################
        # pose_per_individual_videos
        for ind, data in enumerate(fileData):

            if(len(data['hand_right_keypoints_2d'])!= 63):
                print("right - error")
            
            dataOut = {'version':1.3,
                       'people':[{'person_id':[0],
                                  'pose_keypoints_2d':data['pose_keypoints_2d'],
                                  'hand_left_keypoints_2d':data['hand_left_keypoints_2d'],
                                  'face_keypoints_2d':[],
                                  'hand_right_keypoints_2d':data['hand_right_keypoints_2d'],
                                  'face_keypoints_3d':[],
                                  'hand_left_keypoints_3d':[],
                                  'hand_right_keypoints_3d':[]
                                }]
                       }

            json.dumps(dataOut, indent=4, sort_keys=True)
            with open(args.output_Path+'pose_per_individual_videos/'+videoName+'/image_%.5d_keypoints.json'%(ind+1), 'w', encoding='utf-8') as f:
                json.dump(dataOut, f, ensure_ascii=False, indent=4)


json.dumps(bigAsl, indent=4, sort_keys=True)

with open(args.output_Path+"/splits/psl10.json", 'w', encoding='utf-8') as f:
    json.dump(bigAsl, f, ensure_ascii=False, indent=4)


###################################
# Copy needed videos in one folder

origin = args.main_video_path
dest = args.output_Path + "WLASL_VID/"

foldersToLoad = os.listdir(origin)

for folderName in foldersToLoad:
    # if the file have extension then that file will be omited
    if(os.path.splitext(folderName)[1] != ''):
        continue
    
    folderPath = os.listdir(origin+folderName)
    
    for file in folderPath:
        word = file.split('_')[0]
        # To process only topWordList
        if word not in topWordList:
            continue

        copy2(str(origin+folderName+'/'+file), dest)
 

    
