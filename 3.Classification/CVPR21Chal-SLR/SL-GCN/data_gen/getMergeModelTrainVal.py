import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import os

sys.path.extend(['../'])

max_body_true = 1
max_frame = 150
num_channels = 2

char_to_replace = {'Á': 'A',
                   'É': 'E',
                   'Í': 'I',
                   'Ó': 'O',
                   'Ú': 'U'}

def gendata(data_path, label_path, out_path, part='train', config='27'):


    data=[]
    sample_names = []

    meaning =  pd.read_json(label_path)
    meaning = dict(meaning[0])

    df = pd.read_pickle(data_path)

    label_file = pd.read_json(label_path)

    labels = list(df.labels)
    sample_names = list(df.names)

    fp = np.zeros((len(labels), max_frame, 27, num_channels, max_body_true), dtype=np.float32)

    
    for i, skel in enumerate(df.data):

        skel = np.array(skel)

        if skel.shape[0] < max_frame:
            L = skel.shape[0]

            fp[i,:L,:,:,0] = skel

            rest = max_frame - L
            num = int(np.ceil(rest / L))
            pad = np.concatenate([skel for _ in range(num)], 0)[:rest]
            fp[i,L:,:,:,0] = pad

        else:
            L = skel.shape[0]

            fp[i,:,:,:,0] = skel[:max_frame,:,:]

    
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(fp.shape)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    data_path = '../../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-train.pk'
    label_path = "../../../../Data/merged/AEC-PUCP_PSL_DGI156/meaning.json"
    out_folder='../data/sign/'
    points= '1'

    out_path = os.path.join(out_folder, points)

    print(out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    part = "train"
    gendata(data_path, label_path, out_path, part=part, config=points)
    
    print(out_path)
    data_path = '../../../../Data/merged/AEC-PUCP_PSL_DGI156/merge-val.pk'
    part = "val"
    gendata(data_path, label_path, out_path, part=part, config=points)
