import argparse
import pickle
from tqdm import tqdm
import sys
import numpy as np
import pandas as pd
import os

sys.path.extend(['../'])

selected_joints = {
    '59': np.concatenate((np.arange(0,17), np.arange(91,133)), axis=0), #59
    '31': np.concatenate((np.arange(0,11), [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #31
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) #27
}

max_body_true = 1
max_frame = 150
num_channels = 2

char_to_replace = {'Á': 'A',
                   'É': 'E',
                   'Í': 'I',
                   'Ó': 'O',
                   'Ú': 'U'}

def gendata(data_path, label_path, out_path, part='train', config='27'):
    labels = []
    data=[]
    sample_names = []
    selected = selected_joints[config]
    num_joints = len(selected)

    meaning_temp =  pd.read_json('/'.join(label_path.split('/')[:-1]) + '/meaning.json')
    meaning = dict()
    for key, value in meaning_temp[0].items():
        word = key.upper()
        for _key, _value in char_to_replace.items():
            word = word.replace(_key, _value)
        meaning[word]= value

    label_file = pd.read_json(label_path)

    for line in list(label_file[0]):
        
        sample = line.upper()
        for key, value in char_to_replace.items():
            sample = sample.replace(key, value)

        word = "".join(sample.split('_')[:-1])

        sample_names.append(sample)

        data.append(os.path.join(data_path, line + '.mp4.npy'))
        labels.append(meaning[word])

    fp = np.zeros((len(data), max_frame, num_joints, num_channels, max_body_true), dtype=np.float32)

    for i, data_path in enumerate(data):
        # print(sample_names[i])
        skel = np.load(data_path)
        skel = skel[:,selected,:2]
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

    #print(sample_names,labels)
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, labels), f)

    fp = np.transpose(fp, [0, 3, 1, 2, 4])
    print(fp.shape)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':

    part = 'train' #'test' # 'train', 'val'

    parser = argparse.ArgumentParser(description='Sign Data Converter.')
    parser.add_argument('--data_path', default=f'../../data-prepare/data/npy3/{part}') #'train_npy/npy', 'va_npy/npy'
    parser.add_argument('--label_path', default=f"../../../../Data/merged/AEC-PUCP_PSL_DGI156/names-{part}.json") #'../data/sign/27/train_labels.csv') # 'train_labels.csv', 'val_gt.csv', 'test_labels.csv'
    parser.add_argument('--out_folder', default='../data/sign/')
    parser.add_argument('--points', default='27')

    arg = parser.parse_args()

    out_path = os.path.join(arg.out_folder, arg.points)

    print(out_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gendata(
        arg.data_path,
        arg.label_path,
        out_path,
        part=part,
        config=arg.points)
