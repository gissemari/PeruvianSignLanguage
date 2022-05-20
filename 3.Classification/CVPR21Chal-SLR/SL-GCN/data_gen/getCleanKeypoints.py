import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import os
import pandas as pd
import time
from collections import Counter

import cv2

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10],
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
    '1': np.arange(0,27) #np.concatenate((np.arange(0,17) ,np.arange(17,20),np.arange(20,23)),axis=0)
}

max_body_true = 1
max_frame = 150
num_channels = 2#3

def gendata(data_path, out_path, part='train', config='27'):

    labels = []
    data=[]
    sample_names = []
    num_joints = 27
    #label_file = open(label_path, 'r', encoding='utf-8')

    df = pd.read_pickle(data_path)

    data, labels_key, names = list(df.keys())

    labelIndex = list(Counter(df[labels_key]).keys())
    ind = [value for value in range(len(labelIndex))]

    meaning = {lab:idx for idx,lab in zip(ind,labelIndex)}

    listInstance = []
    print(len(df[data]),len(df[data][0]),len(df[data][0][0]),len(df[data][0][0][0]))
    fp = np.zeros((len(df[names]), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)
    sample_names = []
    for pos in range(len(df[names])):

        xCoords = df[data][pos][0]
        yCoords = df[data][pos][1]

        #print(yCoords)

        # [0.5 for _ in range(27)]  
        skel = np.asarray([list(zip(x,y)) for (x, y) in zip(xCoords,yCoords)])

        name = df[names][pos]
        sample_names.append(name)
        label = meaning[df[labels_key][pos]]
        #print(meaning[df[labels_key][pos]])
        labels.append(int(label))
        

        if skel.shape[0] < max_frame:
            L = skel.shape[0] 
            #print(L)
            fp[pos,:L,:,:,0] = skel

            rest = max_frame - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
            fp[pos,L:,:,:,0] = pad

        else:
            L = skel.shape[0]
            #print(L)
            fp[pos,:,:,:,0] = skel[:max_frame,:,:]
        
        #time.sleep(5)
    # print(sample_names,labels)
    print('{}/{}_label.pkl'.format(out_path, part))
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(fp.shape)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    parser.add_argument('--data_path', default='./../../../../Data/AEC_cleaned/data_10_10_24.pk') #'train_npy/npy', 'va_npy/npy'
    parser.add_argument('--out_folder', default='../data/sign/')
    parser.add_argument('--points', default='1') #'27')

    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder, arg.points)
    #print(out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gendata(
        #'./../../../../Data/AEC_cleaned/data_10_10_27-train.pk',
        './../../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-train.pk',
        out_path,
        part="train",
        config=arg.points)

    gendata(
        #'./../../../../Data/AEC_cleaned/data_10_10_27-val.pk',
        './../../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-val.pk',
        out_path,
        part="val",
        config=arg.points)
