# Standard library imports
import argparse
import os
import sys

# Third party imports
import cv2
import pandas as pd
from tqdm import tqdm

# Local imports
sys.path.append('../')
import utils.video as uv

# Title
parser = argparse.ArgumentParser(description='To generate the dictionary that have the dataset metadata')

# File paths
parser.add_argument('--inputPath', type=str, default="../Data/AEC/Videos/SEGMENTED_SIGN/",
                    help='relative path of images input.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')
parser.add_argument('--dict_output', type=str, required=True,
                    help='relative path of scv output set of landmarks.')
parser.add_argument('--label_method', choices=['video_name', 'csv'], default='video_name', help='how to obtain the labels: video_name or csv')

if parser.parse_known_args()[0].label_method == 'csv':
    parser.add_argument('--csv_name', required=True, type=str, help='path to the CSV file containing label mapping')
    parser.add_argument('--dataset', required=True, type=str, help='path to the CSV file containing label mapping')
args = parser.parse_args()


args.inputPath = os.path.normpath(args.inputPath)
dict_output = os.path.normpath(os.sep.join([args.dict_output,"dict.json"]))

df_video_paths = uv.get_list_data(args.inputPath, ['mp4', 'mov'])

if args.label_method == 'csv':
    #label_dict = pd.read_csv(args.csv_path, header=None)
    if args.dataset == 'AUTSL':
        label_dict = pd.read_csv(os.sep.join([args.dict_output,args.csv_name]))
        label_dict = label_dict.set_index('ClassId')['EN'].to_dict()

        train_data = pd.read_csv(os.sep.join([args.dict_output,'train_labels.csv']), header=None)
        val_data   = pd.read_csv(os.sep.join([args.dict_output,'validation_labels.csv']), header=None)
        test_data   = pd.read_csv(os.sep.join([args.dict_output,'test_labels.csv']), header=None)
        data = pd.concat([train_data, val_data, test_data])

    elif args.dataset == 'LSA64':
        label_dict = pd.read_csv(os.sep.join([args.dict_output,args.csv_name]), header=None)
        label_dict = label_dict.set_index(1)[0].to_dict()

print(df_video_paths)
LSP = []
for _num, videoPath in tqdm(enumerate(df_video_paths['path']), desc="Processing"):

    videoPath = os.path.normpath(videoPath)
    video = cv2.VideoCapture(videoPath)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if args.label_method == 'video_name':
        label = os.path.splitext(os.path.basename(videoPath))[0].split('_')[0].upper()
    elif args.label_method == 'csv':
        if args.dataset == 'AUTSL':
            target_name = os.path.splitext(os.path.basename(videoPath))[0].replace('_color','')
            label = data.loc[data[0] == target_name, 1]
            label = label_dict[label.iloc[0]]
        elif args.dataset == 'LSA64':
            target_name = int(os.path.basename(videoPath).split('_')[0])
            label = label_dict[target_name]

    glossInst = {
        "image_dimention": {
            "height": frame_height,
            "witdh": frame_width
        },
        #"keypoints_iD": f"{num}",
        #"image_path": pklImagePath,
        "frame_end": total_frames,
        "frame_start": 1,
        "instance_id": _num,
        "signer_id": -1,
        "fps":fps,
        "source": "LSP",
        "split": "",
        "variation_id": -1,
        "source_video_name": videoPath.replace(args.dict_output, ""),
        #"timestep_vide_name": os.path.splitext(os.path.basename(videoPath))[0]
    }

    # check if there is a gloss asigned with "word"
    glossPos = -1

    for indG, gloss in enumerate(LSP):
        if(gloss["gloss"] == label):
            glossPos = indG

    # in the case word is in the dict
    if glossPos != -1:
        LSP[glossPos]["instances"].append(glossInst)
    else:
        glossDict = {"gloss": str(label),
                    "instances": [glossInst]}
        LSP.append(glossDict)

df = pd.DataFrame(LSP)
df.to_json(dict_output, orient='index', indent=2)