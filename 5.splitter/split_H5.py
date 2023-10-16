# Default import
import argparse
import os
import sys

# Third party imports
import pandas as pd
pd.set_option("display.max_colwidth", 15) 
import h5py
from tqdm import tqdm
import numpy as np

# Local imports
sys.path.append('../')
import utils.video as uv

parser = argparse.ArgumentParser(description='To split the H5 in train/Val')

parser.add_argument('--split', required=True, choices=['LSA64', 'AUTSL'], help='Elija una opción entre Random, LSA64 y AUTSL')
parser.add_argument('--dataset_path', required=True, type=str)
parser.add_argument('--original_list', type=str)
parser.add_argument('--word_list', type=str)
parser.add_argument('--dict_name', type=str)
parser.add_argument('--h5_file', required=True, type=str)

parser.add_argument('--kp_number',choices=[27, 54, 71],type=int, default=54)


parser.add_argument('--no_word_list', action='store_true',)
parser.add_argument('--use_version', action='store_true',)

args = parser.parse_args()

            
def resize_dataset(dataset, value):
    dataset.resize((len(dataset) + 1,))
    dataset[-1] = value

def generate_h5_metadata(h5_file, group_name):
    
    group = h5_file.create_group(group_name)
    
    data = group.create_dataset('data', shape=(0, ), maxshape=(None,), dtype=h5py.special_dtype(vlen='float32'), chunks=True)
    length = group.create_dataset('length', shape=(0,), maxshape=(None,), dtype='int')
    label = group.create_dataset('label', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    class_number = group.create_dataset('class_number', shape=(0,), maxshape=(None,), dtype='int')
    videoName = group.create_dataset('video_name', shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    shape = group.create_dataset('shape', shape=(2,), maxshape=(None,), dtype='int')

    return data, length, label, class_number, videoName, shape

args.dataset_path = os.path.normpath(args.dataset_path)



if args.no_word_list:
    word_list = pd.DataFrame(classes)
    version_file = args.dict_name
else:
    word_list = pd.read_csv(os.sep.join([args.dataset_path, args.word_list]), header=None, na_filter=False)
    version_file = args.word_list

if args.use_version:
    version = int(os.path.splitext(os.path.basename(version_file))[0].split('_')[-1][1:])
    print("Version:",version)



ori_h5_file = h5py.File(os.sep.join([args.dataset_path, args.h5_file]), 'r')

if args.use_version:
    train_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(word_list)}--mediapipe--V{version}-train.hdf5']), 'w')
    val_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(word_list)}--mediapipe--V{version}-val.hdf5']), 'w')
else:
    train_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(classes)}--mediapipe-train.hdf5']), 'w')
    val_h5_file = h5py.File(os.sep.join([args.dataset_path, f'{args.split}--{len(classes)}--mediapipe-val.hdf5']), 'w')

df_keypoints = pd.read_csv('Mapeo landmarks librerias.csv', skiprows=1)

# 29, 54 or 71 points
if args.kp_number == 29:
    df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
elif args.kp_number == 71:
    df_keypoints = df_keypoints[(df_keypoints['Selected 71']=='x' )& (df_keypoints['Key']!='wrist')]
else:
    df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]

idx_keypoints = sorted(df_keypoints['mp_indexInArray'].astype(int).values)


if args.split =="AUTSL":
    
    dict_json = os.sep.join([args.dataset_path, args.dict_name])
    df_video_paths = uv.get_list_from_json_dataset(dict_json)
    classes = df_video_paths['label'].unique()


    train_label = pd.read_csv(os.sep.join([args.dataset_path, 'train_labels.csv']), header=None)
    val_label = pd.read_csv(os.sep.join([args.dataset_path, 'validation_labels.csv']), header=None)
    test_label = pd.read_csv(os.sep.join([args.dataset_path, 'test_labels.csv']), header=None)
    meaning = pd.read_csv(os.sep.join([args.dataset_path, 'SignList_ClassId_TR_EN.csv']))

    meaning = meaning.set_index('ClassId')['EN'].to_dict()

    train_label["class"] = train_label[1].map(meaning)
    val_label["class"] = val_label[1].map(meaning)
    test_label["class"] = test_label[1].map(meaning)

    meaning = {v:k for (k,v) in enumerate(word_list[0])}
    print(meaning)
    train_set = set(train_label[0])
    val_set = set(val_label[0])
    test_set = set(test_label[0])

    if args.no_word_list:
        pass
    else:
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



if args.split == "LSA64":

    ori_meaning = pd.read_csv(os.sep.join([args.dataset_path, args.original_list]), header=None, names=['ID', 'Word'], na_filter=False)
    ori_meaning = ori_meaning.set_index('ID')['Word'].to_dict()

    new_meaning = {v: k for (k, v) in enumerate(word_list[0])}
    
    train_data, train_length, train_label, train_class_number, train_videoName, train_shape = generate_h5_metadata(train_h5_file, 'LSA64')
    val_data, val_length, val_label, val_class_number, val_videoName, val_shape = generate_h5_metadata(val_h5_file, 'LSA64')
    
    _shape = ()
    
    for instance_id in tqdm(ori_h5_file):

        numb = os.path.basename(ori_h5_file[f"{instance_id}"]['video_name'][...].item().decode('utf-8')).split('.')[0].split('_')[-1] 
        label = ori_meaning[int(ori_h5_file[f"{instance_id}"]['label'][...])]
        videoName = ori_h5_file[f"{instance_id}"]['video_name'][...].item().decode('utf-8')
        
        data = ori_h5_file[f"{instance_id}"]['data'][...]
        resized_data = data[:, :, idx_keypoints]
        _shape = resized_data.shape[1:]
        resized_data = resized_data.flatten()

        if numb == '005':
            resize_dataset(val_data, resized_data)
            resize_dataset(val_length, data.shape[0])
            resize_dataset(val_label, label)
            resize_dataset(val_class_number,new_meaning[label] )
            resize_dataset(val_videoName, videoName)
        else:
            resize_dataset(train_data, resized_data)
            resize_dataset(train_length, data.shape[0])
            resize_dataset(train_label, label )
            resize_dataset(train_class_number, new_meaning[label])
            resize_dataset(train_videoName, videoName)
    
    val_shape[:] = _shape
    train_shape[:] = _shape
    


ori_h5_file.close()
train_h5_file.close()
val_h5_file.close()
    