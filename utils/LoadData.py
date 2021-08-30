# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:40:24 2021

@author: Joe
"""
# Standard library imports
import os
from collections import Counter
import random
from random import shuffle

# Third party imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Local imports
#


def getData(is3D=True):

    if is3D:
        dimPath = 'readyToRun/'
    else:
        dimPath = '2D/'

    with open('./Data/Dataset/'+dimPath+'X.data', 'rb') as f:
        X = pickle.load(f)

    with open('./Data/Dataset/'+dimPath+'X_timeSteps.data', 'rb') as f:
        X_timeSteps = pickle.load(f)

    with open('./Data/Dataset/'+dimPath+'Y.data', 'rb') as f:
        Y = pickle.load(f)

    with open('./Data/Dataset/'+dimPath+'weight.data', 'rb') as f:
        weight = pickle.load(f)

    with open('./Data/Dataset/'+dimPath+'Y_meaning.data', 'rb') as f:
        y_meaning = pickle.load(f)

    return X, Y, weight, y_meaning, X_timeSteps


def splitData(x, y, x_timeSteps, split=0.8, timeStepSize=17, leastValue=False,
              balancedTest=False, fixed=True, fixedTest=2,doShuffle=False, augmentation=False):

    # to count repeated targets in y
    targetDict = dict(Counter(y))

    augmentSize = []
    augmentDict = {}

    pivot = targetDict.copy()
    end = targetDict.copy()

    #To select the least value which will be use to split all the data
    if leastValue:

        value = min([val for val in targetDict.values()])
        minValue = int(value*split)

        if(balancedTest):
            minTest = value - minValue - 1
        if(fixed):
            minTest = fixedTest
            minValue = value - fixedTest

    if(augmentation):
        for key, ts in x_timeSteps.items():
            augmentSize.append(sum([val-timeStepSize for val in ts if val > timeStepSize]))

        minAugSize = min(augmentSize)

        augmentDict = {k: minAugSize for k in range(0, len(targetDict.keys()))}

    # to prepare pivot dictionary to use it in the split separator
    for key in targetDict:

        if leastValue:
            pivot[key] = minValue
            if(balancedTest):
                end[key] = minTest
            if(fixed):
                end[key] = fixedTest
        else:
            pivot[key] = int(targetDict[key]*split)

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    if(doShuffle):
        newOrder= list(range(len(y)))
        random.Random(52).shuffle(newOrder)
    count = 1
    countAug = 1

    for pos in newOrder:
        
        # To complete timesteps if it have less than timeStepSize
        for _ in range(timeStepSize - len(x[pos])):
            # Add zeros
            # fileData = np.append(fileData, [np.zeros(len(fileData[0]))], axis=0)

            # repeat the last frame
            x[pos] = np.append(x[pos], [x[pos][-1]], axis=0) 

        #TRAIN
        if(pivot[y[pos]]>=0):
            
            # if have more timesteps than timeStepSize
            if(timeStepSize < len(x[pos])):
                diff = len(x[pos]) - timeStepSize
                if(augmentation and augmentDict[y[pos]]):
                    for start in range(diff):

                        if(not augmentDict[y[pos]]): continue
                        x_train.append(x[pos][start:start+timeStepSize])
                        y_train.append(y[pos])
                        augmentDict[y[pos]] = augmentDict[y[pos]] - 1
                        #print("countAug: %d for %d:%d"%(count,y[pos], augmentDict[y[pos]]))
                        count+=1
                else:
                    start = random.Random(52).choice(range(diff))
                    x_train.append(x[pos][start:start+timeStepSize])
                    y_train.append(y[pos])
                    #print("count: %d"%(count), "at zero")
                    count+=1
            else:
                x_train.append(x[pos])
                y_train.append(y[pos])
                #print("count: %d"%(count))
                count+=1

            pivot[y[pos]] = pivot[y[pos]] - 1
        
        #TEST (balanced)
        elif(leastValue and balancedTest):

            if(end[y[pos]]):
                
                if(timeStepSize < len(x[pos])):
                    diff = len(x[pos]) - timeStepSize
                    if(augmentation and augmentDict[y[pos]]):
                        
                        for start in range(diff):
                            if(not augmentDict[y[pos]]): continue
                            x_test.append(x[pos][start:start+timeStepSize])
                            y_test.append(y[pos])
                            augmentDict[y[pos]] = augmentDict[y[pos]] - 1
                    else:
                        start = random.Random(52).choice(range(diff))
                        x_test.append(x[pos][start:start+timeStepSize])
                        y_test.append(y[pos])
                else:
                    x_test.append(x[pos])
                    y_test.append(y[pos])

                end[y[pos]] = end[y[pos]] - 1

        #TEST
        else:
            if(timeStepSize < len(x[pos])):
                diff = len(x[pos]) - timeStepSize
                start = random.Random(52).choice(range(diff))
                x_test.append(x[pos][start:start+timeStepSize])
                y_test.append(y[pos])
            else:
                x_test.append(x[pos])
                y_test.append(y[pos])

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
    '''
    plt.hist(union, bins=max(union))
    plt.xlabel("Number of time steps")
    plt.ylabel("frequency")
    plt.show

    unionArr = np.asarray(union)
    mean = unionArr.mean()
    std = unionArr.std()

    print("Number of instances:")
    print("-------------------")
    print("2nd deviation range: ", [mean - 2 * std, mean + 2 * std])
    print("Minimun: %d" % min(union))
    print("Maximun: %d" % max(union))
    '''
    return topWords, min(union), 40


def getTopNWordData(nWords, mainFolderPath, minimun=False, is3D=True):

    # return a list of 2D tuple, min and max timestep size
    topNWords, minTimeStep, maxTimeStepAllowed = getTopNWords(
        nWords, mainFolderPath)

    topWordList = [key for (key, value) in topNWords]
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

    x = np.asarray(x)
    y = np.asarray(y)

    if(is3D):
        newX = []
        for instance in x:
            instance = np.reshape(instance, (40, 66), order='C')
            newX.append(instance)
        x = np.asarray(newX)

    return x, y
