# Default import
import argparse
import os
import sys

# Third party imports
import pandas as pd
pd.set_option("display.max_colwidth", 15) 
import h5py
from tqdm import tqdm

# Local imports
sys.path.append('../')
import utils.video as uv

parser = argparse.ArgumentParser(description='To split the H5 in train/Val')

parser.add_argument('--split', required=True, choices=['LSA64', 'AUTSL'], help='Elija una opción entre Random, LSA64 y AUTSL')
parser.add_argument('--dataset_path', required=True, type=str)
parser.add_argument('--word_list', required=True, type=str)
parser.add_argument('--dict_name', required=True, type=str)
parser.add_argument('--h5_file', required=True, type=str)

args = parser.parse_args()

args.dataset_path = os.path.normpath(args.dataset_path)

word_list = pd.read_csv(os.sep.join([args.dataset_path, args.word_list]), header=None)
version = int(os.path.splitext(os.path.basename(args.word_list))[0].split('_')[-1][1:])
print("Version:",version)

dict_json = os.sep.join([args.dataset_path, args.dict_name])

df_video_paths = uv.get_list_from_json_dataset(dict_json)
classes = df_video_paths['label'].unique()

ori_h5_file = h5py.File(os.sep.join([args.dataset_path, args.h5_file]), 'r')
train_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(classes)}--mediapipe--V{version}-train.hdf5']), 'w')
val_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(classes)}--mediapipe--V{version}-val.hdf5']), 'w')

if args.split =="AUTSL":

    train_label = pd.read_csv(os.sep.join([args.dataset_path, 'train_labels.csv']), header=None)
    val_label = pd.read_csv(os.sep.join([args.dataset_path, 'validation_labels.csv']), header=None)
    test_label = pd.read_csv(os.sep.join([args.dataset_path, 'test_labels.csv']), header=None)
    meaning = pd.read_csv(os.sep.join([args.dataset_path, 'SignList_ClassId_TR_EN.csv']))

    meaning = meaning.set_index('ClassId')['EN'].to_dict()

    train_label["class"] = train_label[1].map(meaning)
    val_label["class"] = val_label[1].map(meaning)
    test_label["class"] = test_label[1].map(meaning)

    meaning = {v:k for (k,v) in enumerate(word_list[0])}

    train_set = set(train_label[0])
    val_set = set(val_label[0])
    test_set = set(test_label[0])

    assert not (set(word_list[0]) - set(classes)), "the word list not correspond with the dictionary json"

    print(df_video_paths)
    for index, row in tqdm(df_video_paths.iterrows()):
        instance_name = os.path.basename(row['path']).split('_color')[0]
        if instance_name in train_set:
            ori_h5_file.copy(f"{row['instance_id']}", train_h5_file)
            train_h5_file[f"{row['instance_id']}"]['label'][...] = row['label']
            train_h5_file[f"{row['instance_id']}"]['class_number'] = meaning[row['label']]


        if instance_name in val_set:
            ori_h5_file.copy(f"{row['instance_id']}", val_h5_file)
            val_h5_file[f"{row['instance_id']}"]['label'][...] = row['label']
            val_h5_file[f"{row['instance_id']}"]['class_number'] = meaning[row['label']]

if args.split =="LSA64":

    meaning = {v:k for (k,v) in enumerate(word_list[0])}
    
    for index, row in tqdm(df_video_paths.iterrows()):
        
        numb = os.path.basename(row['path']).split('.')[0].split('_')[-1]

        if numb == '005':
            ori_h5_file.copy(f"{row['instance_id']}", val_h5_file)
            val_h5_file[f"{row['instance_id']}"]['label'][...] = row['label']
            val_h5_file[f"{row['instance_id']}"]['class_number'] = meaning[row['label']]
        else:
            ori_h5_file.copy(f"{row['instance_id']}", train_h5_file)
            train_h5_file[f"{row['instance_id']}"]['label'][...] = row['label']
            train_h5_file[f"{row['instance_id']}"]['class_number'] = meaning[row['label']]

ori_h5_file.close()
train_h5_file.close()
val_h5_file.close()
    