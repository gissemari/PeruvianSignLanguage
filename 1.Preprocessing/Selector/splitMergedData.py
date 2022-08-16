import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection  import train_test_split

def extratXYFromBodyPart(fileData, bodyName, exclusivePoints=[]):

    if exclusivePoints:
        x = [item for pos, item in enumerate(fileData[bodyName]["x"]) if pos in exclusivePoints]
        y = [item for pos, item in enumerate(fileData[bodyName]["y"]) if pos in exclusivePoints]
    else:
        
        x = fileData[bodyName]["x"]
        y = fileData[bodyName]["y"]

    return [[_x ,_y] for _x, _y in zip(x, y)]

def keypointsFormat(fileData):
    
    dataList = []
    
    for pos in range(len(fileData)):
        data = []

        exclusivePoints = [0,11,12,13,14,15,16]
        data = data + extratXYFromBodyPart(fileData[pos],"pose", exclusivePoints)

        exclusivePoints = [0,4,5,8,9,12,13,16,17,20]
        data = data + extratXYFromBodyPart(fileData[pos],"left_hand",exclusivePoints)

        data = data + extratXYFromBodyPart(fileData[pos],"right_hand",exclusivePoints)

        data = np.asarray(data) * 256
        data = data.astype(int)
        dataList.append(data)
    dataList = np.asarray(dataList)
    #dataList = np.moveaxis(dataList, 2, 0)

    return dataList

def getDictData(x_pos, df, jobType):

    pathList = df["paths"]
    labelList = df["labels"]
    nameList = df["names"]

    instanceList = []
    newLabelList = []
    newNameList = []

    tempnamePath = []

    for pos in x_pos:

        tempnamePath.append(pathList[pos].split('/')[-1].split('.')[0])
        
        fileData = pd.read_pickle(pathList[pos])

        data = keypointsFormat(fileData)

        instanceList.append(data)
        newLabelList.append(labelList[pos])
        newNameList.append(nameList[pos])

        #print(pathList[pos].split('/')[-1].split('.')[0], nameList[pos])

    dictData = {
        "data":instanceList,
        "labels":newLabelList,
        "names":newNameList
    }

    temp = pd.DataFrame(tempnamePath)

    temp.to_json(f"Data/merged/AEC-PUCP_PSL_DGI156/names-{jobType}.json",)

    dictData = pd.DataFrame(dictData)
    print(dictData)
    #pickle.dump(dictData,)
    dictData.to_pickle(f"Data/merged/AEC-PUCP_PSL_DGI156/merge-{jobType}.pk")

    return dictData


if __name__ == '__main__':
    basePath = 'Data/merged/AEC-PUCP_PSL_DGI156/merged.pkl'

    df = pd.read_pickle(basePath)
    df = df.T

    pathList = df["paths"]
    labelList = df["labels"]

    x_pos = range(len(pathList))
    X_train, X_test, y_train, y_test = train_test_split(x_pos, labelList, train_size=0.8 , random_state=32, stratify=labelList)

    trainData = getDictData(X_train, df, "train")
    testData = getDictData(X_test, df, "val")

    trainPd = pd.DataFrame(trainData)
    valPd = pd.DataFrame(testData)
