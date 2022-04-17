import argparse
from collections import Counter

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='X and Y of keypoints and image Dataset distribution')
parser.add_argument('--list', action="store_true",help='to shuffle dataset order')
args = parser.parse_args()


wordList = pd.read_csv("./../../Data/list6.csv", header=None)
wordList = list(wordList[0])

df = pd.read_pickle("./../../Data/AEC_cleaned/data_10_10_27.pk")

#print(df)

data, labels, names = list(df.keys())

labelIndex = list(Counter(df[labels]).keys())
labelCount = list(Counter(df[labels]).values())
labelKeyCount = Counter(df[labels])

pivot = {labInd:int(labCnt*0.8) for labInd, labCnt in zip(labelIndex,labelCount)}

#print(len(labelIndex))
#print(len(X_train), len(y_train))
#print(len(X_test), len(y_test))
#print(labelIndex)
ind = [value for value in range(len(labelIndex))]
#print(ind)

meaning = {lab:idx for idx,lab in zip(ind,labelIndex)}
#print(meaning)

trainData = []
trainLabels = []
trainNames = []

valData = []
valLabels = []
valNames = []

for pos in range(len(df[names])):

    label = df[labels][pos]
    name = df[names][pos]
    instanceData = df[data][pos]

    if args.list:

        if label.upper() not in wordList: 
            continue

    if pivot[label] > 0:
        trainData.append(instanceData)
        trainLabels.append(label)
        trainNames.append(name)

        pivot[label] = pivot[label] - 1
    else:
        valData.append(instanceData)
        valLabels.append(label)
        valNames.append(name)

print("Train:",np.asarray(trainData).shape)
print("Val:",np.asarray(valData).shape)

trainDict = {
    data:trainData,
    labels:trainLabels,
    names:trainNames
}
valDict = {
    data:valData,
    labels:valLabels,
    names:valNames
}

#pd.DataFrame.from_records(dt.tolist(), columns=dt.dtype.names)

trainPd = pd.DataFrame(trainDict)
valPd = pd.DataFrame(valDict)

print(trainPd)
print(valPd)

trainPd.to_pickle("./../../Data/AEC_cleaned/data_10_10_27-train.pk")
valPd.to_pickle("./../../Data/AEC_cleaned/data_10_10_27-val.pk")

#df2 = pd.read_pickle("./../../Data/AEC_cleaned/data_10_10_24-train.pk")

#keypoints = np.asarray([np.concatenate((x,y)) for x,y in zip(xCoords,yCoords)])

#print(keypoints.shape)

#allData.append(keypoints)

#print(len(np.asarray(allData)))
