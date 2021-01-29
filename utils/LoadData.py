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

    for folder in foldersToLoad:

        # if the file have extension then that file will be omited
        if(os.path.splitext(folder)[1] != ''):
            continue

        folder = os.listdir(mainFolderPath+folder)

        # wordList => list that acumulate the name of all files of all folders.
        # The acumulated names will have some modification made by
        # file.split('_')[0]
        #
        # file.split('_')[0] => get the name of the file without "_" and
        # its number. then, these names are acumulated in a list that will
        # be added in wordList

        wordList = wordList + [file.split('_')[0] for file in folder]

    # Convert the given list into dictionary
    # the output will be like {'ENTONCES':2,'HOY':3,'MAL':2, ...}
    wordTrends = Counter(wordList)

    # most_common function create a 2D tuple of the N most common words in
    # the dict
    return wordTrends.most_common(nWords)


def getTopNWordData(nWords, mainFolderPath, minimun=False):

    # return a list of 2D tuple
    top20Words = getTopNWords(nWords, mainFolderPath)

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

    for folderName in foldersToLoad:

        # if the file have extension then that file will be omited
        if(os.path.splitext(folderName)[1] != ''):
            continue

        folderList = os.listdir(mainFolderPath+folderName)

        for file in folderList:

            word = file.split('_')[0]

            if word not in topWordList:
                continue

            fileData = pd.read_pickle(mainFolderPath+folderName+'/'+file)

            for data in fileData:
                x.append(data)
                y.append(topWordDict[word])

    # To get the category with the least amount of data. So, all categories
    # have the same amount of data

    if(minimun):
        x, y = ReduceDataToMinimunSize(x, y)

    return x, y
