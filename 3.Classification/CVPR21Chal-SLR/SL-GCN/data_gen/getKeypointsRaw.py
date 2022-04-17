import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import os
import pandas as pd
import math

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
    '1': np.arange(0,133) #np.concatenate((np.arange(0,17) ,np.arange(17,20),np.arange(20,23)),axis=0)
}

max_body_true = 1
max_frame = 150
num_channels = 3

connections = [(5,7),
            (6,8), #1

            (7,9),
            (8,10), #3

            (9,91),#L wrist - L hand
            (10,112), # R wrist - R hand  #5

            (91, 92), #6 #PALM
            (92, 93), (91, 96), (91, 100), (91, 104), (91, 108), (112, 113), (113, 114),
            (112, 114), (112, 117), (112, 121), (112, 125), 
            (112, 129), #18

            (108, 109), #19
            (109, 110), (110, 111), (104, 107), (100, 103), (96, 97), (97, 98), (98, 99), (93, 94),
            (94, 95), (117, 118), (118, 119), (119, 120), (121, 122), (122, 123), (123, 124), (125, 126),
            (126, 127), (127, 128), (129, 130), (130, 131), (131, 132), (114, 115),
            (115, 116), #42
            ]

def gendata(data_paths, out_path, part='train', config='27'):
    labels = []
    data=[]
    sample_names = []
    selected = selected_joints[config]
    num_joints = len(selected)
    label_file = os.listdir(data_paths)

    for line in label_file:
        #print(os.path.join(data_paths, line))
        #sample_names.append(line)
        data.append(os.path.join(data_paths, line))
        # print(line[1])
        #labels.append(int(line[1]))
        # print(labels[-1])

    #fp = np.zeros((len(data), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
    listInstance = []
    for f, data_path in enumerate(data):
        #print("Here",data_path)
        # print(sample_names[i])
        skel = np.load(data_path)
        if not skel.any():
            continue
        skel = skel[:,selected,:]
        kpListOfDict = []
        for timeStep in skel:

            x = timeStep[:,0]
            y = timeStep[:,1]
            outlier = False

            for i, (init, final) in enumerate(connections):

                dist = math.dist([x[init],y[init]], [x[final],y[final]])

                # forearm - Error
                if dist > 110 and i in [2,3]:
                    outlier = True
                    break

                # wrist - Error
                if dist > 35 and i in [4,5]:
                    outlier = True
                    break

                # palm - Error
                if dist > 55 and i in range(6,19):
                    outlier = True
                    break

                # fingers - Error
                if dist > 38   and i in range(19,43): 
                    outlier = True
                    break

            kpDict = {}
            kpDict["x"] = timeStep[:,0]
            kpDict["y"] = timeStep[:,1]
            kpDict["scores"] = timeStep[:,2]
            kpDict["outlier"] = outlier
            kpListOfDict.append(kpDict)

        basePath = os.sep.join([*out_path.split('/')[:2]])
        npyName = data_path.split('/')[-1]

        #fileName = '_'.join(npyName.split('/')[-1].split('_')[:2])

        instanceDict = {}

        instanceDict['label'] = '_'.join(npyName.split('_')[:-1])
        instanceDict['id'] = npyName.split('_')[-1].split('.')[0]
        instanceDict['keypoints']= kpListOfDict
        instanceDict['size'] = len(kpListOfDict)

        listInstance.append(instanceDict)

    df = pd.DataFrame(listInstance)
    print(df)
    print('{}/json/{}.json'.format(basePath, 'allVideosAndPoints-PUCP-3684'))
    df.to_json('{}/json/{}.json'.format(basePath, 'allVideosAndPoints-PUCP-3684'), orient='records')
        #print('{}/json/{}.json'.format(basePath, fileName))
        #df.to_json('{}/json/{}.json'.format(basePath, fileName), orient='records')

    #with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
    #    pickle.dump((sample_names, labels), f)

    #fp = np.transpose(fp, [0, 3, 1, 2, 4])
    #print(fp.shape)
    #np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    parser.add_argument('--data_path', default='../../data-prepare/data/npy3/train') #'train_npy/npy', 'va_npy/npy'
    parser.add_argument('--out_folder', default='../data/sign/')
    parser.add_argument('--points', default='1') #'27')

    part = 'train' #'test' # 'train', 'val'
    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder, arg.points)
    print(out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gendata(
        arg.data_path,
        out_path,
        part=part,
        config=arg.points)
