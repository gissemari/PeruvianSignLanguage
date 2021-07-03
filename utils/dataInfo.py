# -*- coding: utf-8 -*-

# Standard library imports
import argparse
import os
from collections import Counter

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='X and Y Dataset generator')

# Path to folder with videos
parser.add_argument('--main_folder_Path', type=str,
                    default="./Data/Keypoints/pkl_TGCN/Segmented_gestures/")

args = parser.parse_args()

# return a list of 2D tuple
foldersToLoad = os.listdir(args.main_folder_Path)

wordList = []
timeSteps = []
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
        #print(len(fileData))
        if word in timeStepDict:
            timeStepDict[word] = timeStepDict[word] + [len(fileData)]

            timeSteps.append(len(fileData))
            wordList = wordList + [word]
        else:
            timeStepDict.update({word: [len(fileData)]})
            timeSteps.append(len(fileData))

# Convert the given list into dictionary
# the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
wordTrends = Counter(wordList)
wordTrendsList = list(wordTrends.values())

timeStepsArr = np.asarray(timeSteps)

print(timeStepsArr.mean())
print(np.average(timeStepsArr))

plt.hist(timeSteps, bins = 40)
plt.show()

# most_common function create a 2D tuple of the N most common words of
# the Counter

topWords = wordTrends.most_common(10)