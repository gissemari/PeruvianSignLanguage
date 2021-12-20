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
import csv
# Local imports
import utils.video as uv

SEED = 52

parser = argparse.ArgumentParser(description='X and Y of keypoints and image Dataset distribution')

# Path to folder with videos
parser.add_argument('--dict_Path', type=str,
                    default="./Data/Dataset/dict/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/dict/dict.json')

parser.add_argument('--shuffle', action="store_true",help='to shuffle dataset order')

parser.add_argument('--leastValue', action="store_true",help='to shuffle dataset order')

# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./Data/Dataset/readyToRun/",
                    help='relative path of dataset output.' +
                    ' Default: ./Data/Dataset/readyToRun/')

parser.add_argument('--words_File', type=str, default='', #./Data/list5.csv
                    help='relative path of words file' + ' Default: ./Data/')
                    
#parser.add_argument('--wordList', '--names-list', nargs='+', default=[])

# Number of top words
parser.add_argument("--words", type=int, default=10,
                    help="Number of top words")

args = parser.parse_args()
args.shuffle = True


# We get the list of words-gloss from the json file
glossList = pd.read_json(args.dict_Path)

#wordList = ['bien', 'comer', 'cuánto', 'dentro', 'ese', 'fuerte', 'pensar', 'sí', 'tú', 'él']
if args.words_File:
    wordList = pd.read_csv(args.words_File, header=None)
    print("WordList: ", wordList)

#Considering only the top k words in parameter --words
numWords = args.words
#map_object = map(str.upper, wordList.iloc[:,0])
#wordList = list(map_object)[:numWords]
# Saving post or index of the word (that will become the label) of all the instances found in dict.json
wordList = dict(zip(wordList.iloc[:numWords,0], wordList.iloc[:numWords,1]))
print("Number of top words taken: ",numWords, wordList)

temporalList = []
timeStepDict = {}
x = []
y = []


for glossIndex in glossList:

    word = str.upper(glossList[glossIndex]["gloss"])
    print(word)
    if args.words_File != '' and word not in wordList:
        continue
        
    for pos, instance in enumerate(glossList[glossIndex]["instances"]):
    
        # Assign label to the instance
        instanceId = instance["instance_id"]    
        x.append(instanceId)
        y.append(wordList[word])
    
        # Saving time steps, more used in ResNET
        temporalList = temporalList + [word]
        timestep = glossList[glossIndex]["instances"][pos]["frame_end"]

        if word in timeStepDict:
            timeStepDict[word] = timeStepDict[word] + [timestep]
        else:
            timeStepDict.update({word: [timestep]})

# Convert the given list into dictionary to print number of instances ot top words in the dataset
# the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
wordTrends = Counter(temporalList)
print(wordTrends)
topWords = wordTrends.most_common(numWords)
print("\nTOP WORDS")
for pos, top in enumerate(topWords):
    print("%2d)%15s - %d instances"%(pos+1, top[0], top[1]))
print("\nInstance and timesteps size distribution: ")


x_timeSteps = timeStepDict
'''
x_timeSteps = {}
for word, value in topWords:
    x_timeSteps[word] = timeStepDict[word]
    print(word,timeStepDict[word])
    print("Instance size of %d" % len(timeStepDict[word]))
    print()
'''



'''
topWordList = [key for (key, value) in topWords]
# To identify which word corresponds to its label (int: 0, 1, 2, 3, ...)
wordLabels = list(range(len(topWordList)))
# topWordList will be the key list
# wordLabels will be the values list
topWordDict = dict(zip(topWordList, wordLabels))
topWordList = list(topWordDict.keys())
'''


random.Random(SEED).shuffle([])

'''
x = []
y = []
for glossIndex in glossList:
    word = str.upper(glossList[glossIndex]["gloss"])
    if word not in topWordList:
        continue
    for instance in glossList[glossIndex]["instances"]:
        instanceId = instance["instance_id"]
        x.append(instanceId)
        y.append(topWordDict[word])
'''

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


# Balancing the dataset to the min number of instances
if(args.leastValue):
    
    #min value of instances
    minValue = topWords[-1][1]
    
    #pivot = {key:minValue for key in range(len(topWordList))}
    pivot = {key:minValue for key in range(len(wordList))}

    new_x = []
    new_y = []

    # Find minValue instances of each label and save only those.
    for instanceId, label in zip(x, y):
        if(pivot[label]>0):
            new_x.append(instanceId)
            new_y.append(label)

            pivot[label] = pivot[label] - 1
        
    x = new_x
    y = new_y
    
    print("leastValue is active, so all the distribution have the size of the least instance size:", topWords[-1])


# to get weights for all topwords selected
dict_weight = Counter(y)

total_weight = max([w for w in dict_weight.values()])
weight = [dict_weight[w]/total_weight for w in range(numWords)]

# to switch key and values to have y number meaning
#y_meaning = {_y: _x for _x, _y in topWordDict.items()}
y_meaning = {_y: _x for _x, _y in wordList.items()}

uv.createFolder(args.output_Path)

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
