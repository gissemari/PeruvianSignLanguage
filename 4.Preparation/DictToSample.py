# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:35:07 2021

@author: Joe
"""
# Standard library imports
import argparse
import os
import random
from collections import Counter

# Third party imports
import pandas as pd
import pickle as pkl
import numpy as np

# Local imports

SEED = 52

parser = argparse.ArgumentParser(description='X and Y of keypoints and image Dataset distribution')


# Path to folder with videos
parser.add_argument('--dict_Path', type=str,
                    default="./Data/Dataset/dict/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Keypoints/pkl/Segmented_gestures/')

parser.add_argument('--shuffle', action="store_true",
                    help='to shuffle dataset order')

parser.add_argument('--leastValue', action="store_true",
                    help='to shuffle dataset order')

# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./Data/Dataset/readyToRun/",
                    help='relative path of dataset output.' +
                    ' Default: ./Data/Dataset/readyToRun/')

parser.add_argument('--keypoints_Path', type=str,
                    default="./Data/Dataset/keypoints/",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/keypoints/')

parser.add_argument('--image_Path', type=str,
                    default="./Data/Dataset/img/",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/keypoints/')

parser.add_argument("--timeStepSize", type=int, default=17,
                    help="Number of timestep size you want")

parser.add_argument('--wordList', '--names-list', nargs='+', default=[])

# Number of top words
parser.add_argument("--words", type=int, default=10,
                    help="Number of top words")

args = parser.parse_args()


if(args.wordList):
    print("WordList: ", args.wordList)
    args.words = len(args.wordList)
else:
    print("Number of words: ",args.words)

temporalList = []
timeStepDict = {}

glossList = pd.read_json(args.dict_Path)

for glossIndex in glossList:

    word = glossList[glossIndex]["gloss"]

    if args.wordList and word not in args.wordList:
        continue

    for pos, _ in enumerate(glossList[glossIndex]["instances"]):
        temporalList = temporalList + [word]
        timestep = glossList[glossIndex]["instances"][pos]["frame_end"]

        if word in timeStepDict:
            timeStepDict[word] = timeStepDict[word] + [timestep]
        else:
            timeStepDict.update({word: [timestep]})

# Convert the given list into dictionary
# the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
wordTrends = Counter(temporalList)

topWords = wordTrends.most_common(args.words)

print("\nTOP WORDS")
for pos, top in enumerate(topWords):
    print("%2d)%15s - %d instances"%(pos+1, top[0], top[1]))

print("\nTimesteps size distribution (with timeStepSize as default): ")
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

topWordList = list(topWordDict.keys())

random.Random(SEED).shuffle([])

x = []
y = []

for glossIndex in glossList:

    word = glossList[glossIndex]["gloss"]
    
    if word not in topWordList:
        continue
    for instance in glossList[glossIndex]["instances"]:

        instanceId = instance["instance_id"]
        
        x.append(instanceId)
        
        y.append(topWordDict[word])



if(args.shuffle):
    newOrder = list(range(len(y)))
    random.Random(SEED).shuffle(newOrder)
    
    new_x = []
    new_y = []

    for pos in newOrder:
        new_x.append(x[pos])
        new_y.append(y[pos])

    x = new_x
    y = new_y

if(args.leastValue):
    
    minValue = topWords[-1][1]
    
    pivot = {key:minValue for key in range(len(topWordList))}

    new_x = []
    new_y = []

    for instanceId, label in zip(x, y):
        if(pivot[label]>0):
            new_x.append(instanceId)
            new_y.append(label)

            pivot[label] = pivot[label] - 1
        
    x = new_x
    y = new_y
    
if(args.timeStepSize > 1):
    for pos in x:
        dataKP = pd.read_pickle(args.keypoints_Path + str(pos) + '.pkl')
        dataIMG = pd.read_pickle(args.image_Path + str(pos) + '.pkl')

        if len(dataKP) == args.timeStepSize:
            continue
        # To complete the number of timesteps if it is less than requiered
        elif len(dataKP) < args.timeStepSize:
    
            for _ in range(args.timeStepSize - len(dataKP)):
                dataKP = np.append(dataKP, [dataKP[-1]], axis=0)
                dataIMG = np.append(dataIMG, [dataIMG[-1]], axis=0)
        # More than the number of timesteps
        else:
    
            toSkip = len(dataKP) - args.timeStepSize
            interval = len(dataKP) // toSkip
    
            # Generate an interval of index
            a = [val for val in range(0, len(dataKP)) if val % interval == 0]
    
            # from the list of index, we erase only the number of index we want to skip
            dataKP = np.delete(dataKP, a[-toSkip:], axis=0)
            dataIMG = np.delete(dataIMG, a[-toSkip:], axis=0)

        with open(args.keypoints_Path + str(pos) + '.pkl', 'wb') as pickle_file:
                pkl.dump(dataKP, pickle_file)

        with open(args.image_Path + str(pos) + '.pkl', 'wb') as pickle_file:
                pkl.dump(dataIMG, pickle_file)
    print()
    print("TimeStepSize of %d -> Now, All timesteps have size %d" % (args.timeStepSize, args.timeStepSize))
        


# to get weights for all topwords selected
dict_weight = Counter(y)

total_weight = max([w for w in dict_weight.values()])
weight = [dict_weight[w]/total_weight for w in range(args.words)]

# to switch key and values to have y number meaning
y_meaning = {_y: _x for _x, _y in topWordDict.items()}

with open(args.output_Path+'X.data', 'wb') as f:
    pkl.dump(x, f)

with open(args.output_Path+'Y.data', 'wb') as f:
    pkl.dump(y, f)

with open(args.output_Path+'weight.data', 'wb') as f:
    pkl.dump(weight, f)

with open(args.output_Path+'Y_meaning.data', 'wb') as f:
    pkl.dump(y_meaning, f)

with open(args.output_Path+'X_timeSteps.data', 'wb') as f:
    pkl.dump(x_timeSteps, f)