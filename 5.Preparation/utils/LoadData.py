# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:40:24 2021

@author: Joe
"""
# Standard library imports
import os
from collections import Counter
import random


# Third party imports
import pandas as pd
import numpy as np
import pickle

# Local imports
#

def getData(path):

    with open(path+'X.data', 'rb') as f:
        X = pickle.load(f)

    with open(path+'X_timeSteps.data', 'rb') as f:
        X_timeSteps = pickle.load(f)

    with open(path+'Y.data', 'rb') as f:
        Y = pickle.load(f)

    with open(path+'weight.data', 'rb') as f:
        weight = pickle.load(f)

    with open(path+'Y_meaning.data', 'rb') as f:
        y_meaning = pickle.load(f)

    return X, Y, weight, y_meaning, X_timeSteps


def timeStepFormat(fileData, timeStepSize):

    # if it has the desired size
    if len(fileData) == timeStepSize:
        return fileData
    
    # To complete the number of timesteps if it is less than requiered
    elif len(fileData) < timeStepSize:
        for _ in range(timeStepSize - len(fileData)):
            fileData = np.append(fileData, [fileData[-1]], axis=0)
        return fileData
    # More than the number of timesteps
    else:

        toSkip = len(fileData) - timeStepSize
        interval = len(fileData) // toSkip

        # Generate an interval of index
        a = [val for val in range(0, len(fileData)) if val % interval == 0]

        # from the list of index, we erase only the number of index we want to skip
        fileData = np.delete(fileData, a[-toSkip:], axis=0)

        return fileData

def extratXYFromBodyPart(fileData, bodyName, exclusivePoints=[]):

    if exclusivePoints:
        x = [item for pos, item in enumerate(fileData[bodyName]["x"]) if pos in exclusivePoints]
        y = [item for pos, item in enumerate(fileData[bodyName]["y"]) if pos in exclusivePoints]
    else:
        
        x = fileData[bodyName]["x"]
        y = fileData[bodyName]["y"]

    return [item for sublist in zip(x,y) for item in sublist][:-1]


def getXInfo(src, pos, timeStepSize=-1):
    fileData = pd.read_pickle(src + str(pos) + '.pkl')

    if timeStepSize == -1:
        return fileData
    
    fileData = timeStepFormat(fileData, timeStepSize)
    return fileData


def keypointsFormat(fileData, bodyPart):
    
    dataList = []
    
    for pos in range(len(fileData)):
        data = []

        for bodyName in bodyPart:
            if(bodyName == "pose"):
                data = data + extratXYFromBodyPart(fileData[pos],"pose")
            elif(bodyName == "hands"):
                data = data + extratXYFromBodyPart(fileData[pos],"left_hand")
                data = data + extratXYFromBodyPart(fileData[pos],"right_hand")

            elif(bodyName == "face"):

                nose_points = [1,5,6,218,438]
                mouth_points = [78,191,80,81,82,13,312,311,310,415,308,
                                95,88,178,87,14,317,402,318,324,
                                61,185,40,39,37,0,267,269,270,409,291,
                                146,91,181,84,17,314,405,321,375]
                #mouth_points = [0,37,39,40,61,185,267,269,270,291,409, 
                #                12,38,41,42,62,183,268,271,272,292,407,
                #                15,86,89,96,179,316,319,325,403,
                #                17,84,91,146,181,314,321,375,405]
                left_eyes_points = [33,133,157,158,159,160,161,173,246,
                                    7,144,145,153,154,155,163]
                left_eyebrow_points = [63,66,70,105,107]
                                       #46,52,53,55,65]
                right_eyes_points = [263,362,384,385,386,387,388,398,466,
                                     249,373,374,380,381,382,390]
                right_eyebrow_points = [293,296,300,334,336]
                                        #276,282,283,285,295]
  
                #There are 97 points
                exclusivePoints = nose_points
                exclusivePoints = exclusivePoints + mouth_points
                exclusivePoints = exclusivePoints + left_eyes_points
                exclusivePoints = exclusivePoints + left_eyebrow_points
                exclusivePoints = exclusivePoints + right_eyes_points
                exclusivePoints = exclusivePoints + right_eyebrow_points
                
                data = data + extratXYFromBodyPart(fileData[pos],"face",exclusivePoints)
        dataList.append(np.asarray(data))
    return np.asarray(dataList)

def getKeypointsfromIdList(src, idList, bodyPart=["pose","face","hands"] , timeStepSize=-1):
    
    data = []
    
    for pos in idList:

        videoList = os.listdir(src)

        for video in videoList:
            pklList = os.listdir(src.replace('/',os.sep)+video)
            for pkl in pklList:

                if str(pos) in pkl.split('_')[1].split('.')[0]:
                    fileData = getXInfo(src.replace('/',os.sep)+video+os.sep, pkl.split('.')[0], timeStepSize)
                    data.append(keypointsFormat(fileData, bodyPart))
                    break

    return np.asarray(data)

def getImagefromIdList(src, idList, timeStepSize=-1):
    
    data = []
    
    for pos in idList:
        fileData = pd.read_pickle(src + str(pos) + '.pkl')

        #without format timeStep
        if timeStepSize == -1:
            data.append(fileData)
            continue

        fileData = timeStepFormat(fileData, timeStepSize)

        data.append(fileData)
    
    return data

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
