# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 14:04:44 2021

@author: Joe
"""
# Standard library imports
import argparse
import os
from collections import Counter

# Third party imports
import pandas as pd
import numpy as np
import pickle as pkl

# Local imports
import utils.video as uv  # for folder creation

parser = argparse.ArgumentParser(description='X and Y Dataset distribution')

# Path to folder with videos
parser.add_argument('--main_folder_Path', type=str,
                    default="./Data/Keypoints/pkl/Segmented_gestures/",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Keypoints/pkl/Segmented_gestures/')

# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./Data/Dataset/toReshape/",
                    help='relative path of dataset output.' +
                    ' Default: ./Data/Dataset/toReshape/')

parser.add_argument('--wordList', '--names-list', nargs='+', default=[])

# Number of top words
parser.add_argument("--words", type=int, default=10,
                    help="Number of top words")

args = parser.parse_args()

if(args.wordList):
    print("WordList: ", args.wordList)
else:
    print("Number of words: ",args.words)

# return a list of 2D tuple
foldersToLoad = os.listdir(args.main_folder_Path)

temporalList = []
timeStepDict = {}

for videoFolderName in foldersToLoad:

    # if the file have extension then that file will be omited
    if(os.path.splitext(videoFolderName)[1] != ''):
        continue

    folder = os.listdir(args.main_folder_Path+videoFolderName)

    # temporalList => list that acumulate the name of all files of all folders.
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

        if args.wordList and word not in args.wordList:
            continue

        fileData = pd.read_pickle(args.main_folder_Path+videoFolderName+'/'+file)
        
        if word in timeStepDict:
            timeStepDict[word] = timeStepDict[word] + [len(fileData)]
            temporalList = temporalList + [word]
        else:
            timeStepDict.update({word: [len(fileData)]})

#timeStepRank = {}

#for key in timeStepDict.keys():
#    timeStepRank[key] = sum(timeStepDict[key])

# Convert the given list into dictionary
# the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
wordTrends = Counter(temporalList)

#sortedTimeStepRank = sorted(timeStepRank.items(), key=lambda x: x[1], reverse=True)
#print(sortedTimeStepRank[0:10])

#print([(k,wordTrends[k]) for (k, v) in sortedTimeStepRank[0:10]])
# most_common function create a 2D tuple of the N most common words of
# the Counter

topWords = wordTrends.most_common(args.words)

print("\nTOP WORDS")
for pos, top in enumerate(topWords):
    print("%2d)%15s - %d instances"%(pos+1, top[0], top[1]))

print("\nTimesteps size distribution : ")
print()
x_timeSteps = {}
for word, value in topWords:
    x_timeSteps[word] = timeStepDict[word]
    print(word,timeStepDict[word])
    print()

topWordList = [key for (key, value) in topWords]

wordLabels = list(range(len(topWordList)))

# To identify which word corresponds to its label (int: 1, 2, 3, ...)
#
# topWordList will be the key list
# wordLabels will be the values list
topWordDict = dict(zip(topWordList, wordLabels))

x = []
y = []
y_oneHot = []

foldersToLoad = os.listdir(args.main_folder_Path)

# folders
for videoFolderName in foldersToLoad:

    # if the file have extension then that file will be omited
    if(os.path.splitext(videoFolderName)[1] != ''):
        continue

    folderList = os.listdir(args.main_folder_Path+videoFolderName)

    # file => instance
    for file in folderList:

        word = file.split('_')[0]

        # To process only topWordList
        if word not in topWordList:
            continue

        fileData = pd.read_pickle(args.main_folder_Path+videoFolderName+'/'+file)


        oneHot = np.zeros(args.words)
        oneHot[topWordDict[word]] = 1

        x = x + [list(fileData)]
        y = y + [topWordDict[word]]
        y_oneHot = y_oneHot + [oneHot]

x = np.asarray(x, dtype="object")
print("X shape: ", x.shape)
y = np.asarray(y)
print("Y shape: ", y.shape)
y_oneHot = np.asarray(y_oneHot)
print("Y (one hot) shape: ", y_oneHot.shape)

# to switch key and values to have y number meaning
y_meaning = {_y: _x for _x, _y in topWordDict.items()}

# to get weights for all topwords selected
dict_weight = Counter(y)

total_weight = max([w for w in dict_weight.values()])
weight = [dict_weight[w]/total_weight for w in range(args.words)]

weight = np.asarray(weight)
print("weight shape: ", weight.shape)
print("\nData saved in: ", args.output_Path)

uv.createFolder(args.output_Path)

with open(args.output_Path+'X.data', 'wb') as f:
    pkl.dump(x, f)

with open(args.output_Path+'Y.data', 'wb') as f:
    pkl.dump(y, f)

with open(args.output_Path+'weight.data', 'wb') as f:
    pkl.dump(weight, f)

with open(args.output_Path+'Y_oneHot.data', 'wb') as f:
    pkl.dump(y_oneHot, f)

with open(args.output_Path+'Y_meaning.data', 'wb') as f:
    pkl.dump(y_meaning, f)

with open(args.output_Path+'X_timeSteps.data', 'wb') as f:
    pkl.dump(x_timeSteps, f)
