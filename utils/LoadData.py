# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:40:24 2021

@author: Joe
"""
# Standard library imports
import os
from collections import Counter

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Local imports


def splitData(x, y, split=0.8):

    # to count repeated targets in y
    targetDict = dict(Counter(y))

    pivot = targetDict.copy()

    for key in targetDict:

        pivot[key] = int(targetDict[key]*split)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    for index, key in enumerate(y):

        if(pivot[key]):
            x_train.append(x[index])
            y_train.append(y[index])

            pivot[key] = pivot[key] - 1
        else:
            x_test.append(x[index])
            y_test.append(y[index])

    return x_train, y_train, x_test, y_test


def ReduceDataToMinimunSize(x, y):

    # to count repeated targets in y
    targetDict = dict(Counter(y))

    # get the least mount of data of all categories
    minimun = min(targetDict.values())

    newX = []
    newY = []

    # to set a counter for all categories to the minimun data
    reverseCounter = dict.fromkeys(targetDict, minimun)

    for index, key in enumerate(y):

        # if reverseCounter is not zero(key)
        if reverseCounter[key]:

            newX.append(x[index])
            newY.append(y[index])

            # to do reverse counter from minimun to zero
            reverseCounter[key] = reverseCounter[key] - 1

    return newX, newY


# getlist of 2D tuple of the most common N words of a filename in a group
# of folders
def getTopNWords(nWords, mainFolderPath):

    foldersToLoad = os.listdir(mainFolderPath)

    wordList = []
    timeStepDict = {}

    for folderName in foldersToLoad:

        # if the file have extension then that file will be omited
        if(os.path.splitext(folderName)[1] != ''):
            continue

        folder = os.listdir(mainFolderPath+folderName)

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

            fileData = pd.read_pickle(mainFolderPath+folderName+'/'+file)

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
    topWords = wordTrends.most_common(nWords)

    # to acumulate timesteps to do statistic
    union = []

    for word, value in topWords:
        union = union + timeStepDict[word]

    plt.hist(union, bins=max(union))
    plt.xlabel("Number of time steps")
    plt.ylabel("frequency")
    plt.show

    unionArr = np.asarray(union)
    mean = unionArr.mean()
    std = unionArr.std()

    '''
    print("Number of instances:")
    print("-------------------")
    print("2nd deviation range: ", [mean - 2 * std, mean + 2 * std])
    print("Minimun: %d" % min(union))
    print("Maximun: %d" % max(union))
    '''
    return topWords, int(mean + 2 * std)+1


def getTopNWordData(nWords, mainFolderPath, minimun=False):

    # return a list of 2D tuple
    top20Words, maxTimeStepAllowed = getTopNWords(nWords, mainFolderPath)

    topWordList = [key for (key, value) in top20Words]
    wordLabels = list(range(len(topWordList)))

    # To identify which word corresponds to its label (int: 1, 2, 3, ...)
    #
    # topWordList will be the key list
    # wordLabels will be the values list
    topWordDict = dict(zip(topWordList, wordLabels))

    x = []
    y = []

    foldersToLoad = os.listdir(mainFolderPath)

    # folders
    for folderName in foldersToLoad:

        # if the file have extension then that file will be omited
        if(os.path.splitext(folderName)[1] != ''):
            continue

        folderList = os.listdir(mainFolderPath+folderName)

        # instances
        for file in folderList:

            word = file.split('_')[0]

            if word not in topWordList:
                continue

            fileData = pd.read_pickle(mainFolderPath+folderName+'/'+file)

            timeStepSize = len(fileData)

            if timeStepSize > maxTimeStepAllowed:

                continue
            fileData = fileData.flatten()
            for _ in range(maxTimeStepAllowed - timeStepSize):

                fileData = np.append(fileData, [np.zeros(66)])

            x = x + [list(fileData)]
            y = y + [topWordDict[word]]

    return x, y, maxTimeStepAllowed
