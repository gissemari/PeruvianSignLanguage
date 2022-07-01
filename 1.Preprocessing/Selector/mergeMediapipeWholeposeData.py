import pickle
import math
import sys
import os

import pandas as pd
import numpy as np

from splitMergedData import keypointsFormat 
from sklearn.model_selection  import train_test_split

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
    '1': np.arange(0,133) #np.concatenate((np.arange(0,17) ,np.arange(17,20),np.arange(20,23)),axis=0)
}

connections = [(5,7),
            (6,8), #1

            (7,9),
            (8,10), #3


            (9,91), # left hands #4
            (91, 96), (91, 100), (91, 104), (91, 108), (91, 93), 
            (91, 92), (92, 93), (93, 94), (94, 95),
            (96, 97), (97, 98), (98, 99), 
            (100, 101), (101, 102), (102, 103),
            (104, 105), (105, 106), (106, 107),
            (108, 109), (109, 110), (110, 111), #25
            
            (10,112), #right Hands #26
            (112, 117), (112, 121), (112, 125), (112, 129), (112, 114), 
            (112, 113), (113, 114), (114, 115), (115, 116),
            (117, 118), (118, 119), (119, 120), 
            (121, 122), (122, 123), (123, 124), 
            (125, 126), (126, 127), (127, 128),
            (129, 130), (130, 131), (131, 132), #47
            ]

def genWholePosedata(data_paths, whiteList, bannedList, out_path, meaning, config='27'):

    labels = []
    data=[]
    sample_names = []
    selected = selected_joints[config]
    num_joints = len(selected)
    label_file = os.listdir(data_paths)

    for line in label_file:
        
        word = line.split('_')[0].upper()

        if word in bannedList:
            continue

        if word not in whiteList:
            continue

        data.append(os.path.join(data_paths, line))

    listInstance = []

    for f, data_path in enumerate(data):

        skel = np.load(data_path)
        if not skel.any():
            continue

        skel = skel[:,selected,:2]

        basePath = os.sep.join([*out_path.split('/')[:2]])
        npyName = data_path.split('/')[-1]

        timestepError = []

        for timeStep in skel:

            x = timeStep[:,0]
            y = timeStep[:,1]
            outlier = []

            for i, (init, final) in enumerate(connections):

                dist = math.dist([x[init],y[init]], [x[final],y[final]])

                # pose - Error
                if dist > 110 and i in [2,3] and 'pose' not in outlier:
                    outlier.append('pose')                 

                # left - Error
                if dist > 55 and i in range(4,26) and 'left' not in outlier:
                    outlier.append("left")
                    

                # fingers - Error
                if dist > 38   and i in range(26,48) and 'right' not in outlier: 
                    outlier.append("right")
                    
            timestepError.append(outlier)

        skel = skel[:,selected_joints['27'],:]

        assert len(skel[0]) == 27

        word = "".join(npyName.split('_')[:-1]).split('.')[0].upper()

        instanceDict = {}
        instanceDict['words'] = word
        instanceDict['timestepError'] = timestepError
        instanceDict['data']= skel
        instanceDict['labels'] = meaning[word]
        instanceDict['names'] = npyName.split(".")[0]

        listInstance.append(instanceDict)

    df = pd.DataFrame(listInstance)

    return df

def genMediapipedata(whiteList, bannedList, meaning):
    dataset = ["AEC", "PUCP_PSL_DGI156"]

    listInstance = []

    for opt in dataset:

        dictPath = "Data/"+opt+'/dict.json'

        gloss = pd.read_json(dictPath)

        for glossIndex in gloss:

            word = gloss[glossIndex]["gloss"].upper()
        
            for inst in  gloss[glossIndex]["instances"]:

                if not word in whiteList:
                    continue

                timestepError = []

                fileData = pd.read_pickle(inst["keypoints_path"])
                
                data = keypointsFormat(fileData)
                data = np.moveaxis(data, 1, 0)
                data = np.moveaxis(data, 1, 2)

                for timestep in data:
                    assert len(timestep) == 27
                    error = []
                    for pos, point in enumerate(timestep):
                        if pos in range(0,7) and "pose" in error:
                            continue
                        if pos in range(7,17) and "left" in error:
                            continue
                        if pos in range(17,27) and "right" in error:
                            continue

                        if point[0] < 0 or point[0] > 256 or point[1] < 0 or point[1] > 256 or (point[0] == 256 and point[1]== 256):
                        #if (point[0] == 256 and point[1]== 256):
                            if pos in range(0,7):
                                error.append("pose")
                            if pos in range(7,17):
                                error.append("left")
                            if pos in range(17,27):
                                error.append("right")

                    timestepError.append(error)

                instanceDict = {}
                instanceDict['words'] = word
                instanceDict['timestepError'] = timestepError
                instanceDict['labels'] = meaning[word]
                instanceDict['data']= data
                instanceDict['dataset'] = opt
                instanceDict['names'] = inst["unique_name"]
                listInstance.append(instanceDict)


    df = pd.DataFrame(listInstance)

    return(df)

def improveDatasets(mpData,wbData, meaning):

    mpData = mpData.sort_values("names")
    wbData = wbData.sort_values("names")

    for mpNames, wbNames in zip(mpData.names, wbData.names):
        assert mpNames == wbNames

    listInstance = []

    for word, dataset, name, mpData, wbData, mpError, wbError, index in zip(mpData.words, mpData.dataset, wbData.names, mpData.data, wbData.data, mpData.timestepError, wbData.timestepError, mpData.index):
        timestepList = []
        for mpTimeStep, wbTimeStep, mpTsError, wbTsError in zip(mpData, wbData, mpError, wbError):

            # LEFT HAND
            if "left" in mpTsError and "left" in wbTsError:
                left = wbTimeStep[range(7,17)]

            elif "left" in mpTsError:
                left = wbTimeStep[range(7,17)]

            elif "left" in wbTsError:
                left = mpTimeStep[range(7,17)]

            else:
                left = wbTimeStep[range(7,17)]

            # RIGHT HAND
            if "right" in mpTsError and "right" in wbTsError:
                right = wbTimeStep[range(17,27)]

            elif "right" in mpTsError:
                right = wbTimeStep[range(17,27)]

            elif "right" in wbTsError:
                right = mpTimeStep[range(17,27)]

            else:
                right = wbTimeStep[range(17,27)]

            # POSE
            if "pose" in mpTsError and "pose" in wbTsError:
                pose = wbTimeStep[range(0,7)]
                #pose = np.array(mpTimeStep[range(0,5)]) + np.array(wbTimeStep[range(0,5)])
                #pose = (pose / 2)
                #pose = pose.astype(int).tolist()
                #pose.append(left[0].astype(int).tolist())
                #pose.append(right[0].astype(int).tolist())
                #pose = np.array(pose)

            elif "pose" in mpTsError:
                pose = wbTimeStep[range(0,7)]

            elif "pose" in wbTsError:
                pose = mpTimeStep[range(0,7)]
                
            else:
                pose = wbTimeStep[range(0,7)]

            #if(wbTsError):
            #    continue
            pose = pose.astype(int).tolist()
            left = left.astype(int).tolist()
            right = right.astype(int).tolist()

            data = pose + left + right
            timestepList.append(data)

        if timestepList:
            instanceDict = {}
            instanceDict['words'] = word
            instanceDict['data']= timestepList
            instanceDict['dataset'] = dataset
            instanceDict['names'] = name
            instanceDict['labels'] = meaning[word]
            instanceDict['index'] = index
            listInstance.append(instanceDict)

    df = pd.DataFrame(listInstance)
    df = df.sort_values("index")

    #df.dropna(subset = ["column2"], inplace=True)
    return df

def getDictData(posList, data, jobType):

    dataList = data.data
    labelList = data.labels
    nameList = data.names

    instanceList = dataList[posList]
    newLabelList = labelList[posList]
    newNameList = nameList[posList]

    dictData = {
        "data":instanceList,
        "labels":newLabelList,
        "names":newNameList
    }

    dictData = pd.DataFrame(dictData)

    print(dictData)

    dictData.to_pickle(f"Data/merged/AEC-PUCP_PSL_DGI156/merge-{jobType}.pk")

    return dictData


def splitData(data):
    labels = data.labels

    x_pos = range(len(labels))
    X_train, X_test, y_train, y_test = train_test_split(x_pos, list(labels), train_size=0.8 , random_state=42, stratify=list(labels))

    trainData = getDictData(X_train, data, "train")
    testData = getDictData(X_test, data, "val")

if __name__ == '__main__':

    wholeposeDataPath = "3.Classification/CVPR21Chal-SLR/data-prepare/data/npy3/other/"

    out_path = "Data/merged/AEC-PUCP_PSL_DGI156"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    whiteList = ['PENSAR', 'CÓMO', 'NO', 'CASA', 'SÍ', 'UNO', 'MAMÁ', 'MUJER', 'PORCENTAJE', 'PROTEÍNA', 'CUÁNTO', 'ESE', 'COMER', 'HOMBRE', 'CAMINAR']
    #bannedList = ["???", "","YA","QUÉ","QUÉ?","BIEN","DOS","","AHÍ","LUEGO","YO","ÉL","TÚ"]
    bannedList = []

    meaning = {word:pos for pos, word in enumerate(whiteList)}

    wholeposeData = genWholePosedata(wholeposeDataPath, whiteList, bannedList, out_path, meaning, config="1")

    mediapipeData = genMediapipedata(whiteList, bannedList, meaning)

    newData = improveDatasets(mediapipeData, wholeposeData, meaning)

    splitData(newData)

    print(pd.DataFrame.from_dict(meaning, orient='index'))

    meaning = {pos:word for word, pos in meaning.items()}

    df_meaning = pd.DataFrame.from_dict(meaning, orient='index')

    df_meaning.to_json('Data/merged/AEC-PUCP_PSL_DGI156/meaning.json')

