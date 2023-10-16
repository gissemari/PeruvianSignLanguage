import argparse
import os
import sys
import cv2
import pandas as pd
from tqdm import tqdm
from os.path import join, normpath, basename, splitext

sys.path.append('../')
import utils.video as uv

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a dictionary with dataset metadata')

    # File paths
    parser.add_argument('--inputVideoPath', type=str, help='Relative path of images input')
    parser.add_argument('--dict_output', type=str, required=True, help='Relative path of scv output set of landmarks.')
    parser.add_argument('--label_method', choices=['video_name', 'csv'], default='video_name',
                        help='How to obtain the labels: video_name or csv')

    if parser.parse_known_args()[0].label_method == 'csv':
        parser.add_argument('--csv_name', required=True, type=str, help='Path to the CSV file containing label mapping')
        parser.add_argument('--dataset', required=True, type=str, help='Path to the CSV file containing label mapping')

    return parser.parse_args()

def read_label_csv(csv_path):
    label_dict = pd.read_csv(csv_path, header=None, names=['ID', 'Word'], na_filter=False)
    return label_dict.set_index('ID')['Word'].to_dict()

def process_video(args, label_dict, LSP, video_path, num):
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if args.label_method == 'video_name':
        label = splitext(basename(video_path))[0].split('_')[0].upper()
    elif args.label_method == 'csv':
        if args.dataset == 'AUTSL':
            target_name = splitext(basename(video_path))[0].replace('_color', '')
            label = data.loc[data[0] == target_name, 1]
            label = label_dict[label.iloc[0]]
        elif args.dataset == 'LSA64':
            target_name = int(basename(video_path).split('_')[0])
            label = label_dict[target_name]

    gloss_inst = {
        "image_dimention": {"height": frame_height, "width": frame_width},
        "frame_end": total_frames,
        "frame_start": 1,
        "instance_id": num,
        "signer_id": -1,
        "fps": fps,
        "source": "LSP",
        "split": "",
        "variation_id": -1,
        "source_video_name": video_path.replace(args.dict_output, "")
    }

    gloss_pos = next((indG for indG, gloss in enumerate(LSP) if gloss["gloss"] == label), -1)

    if gloss_pos != -1:
        LSP[gloss_pos]["instances"].append(gloss_inst)
    else:
        gloss_dict = {"gloss": str(label), "instances": [gloss_inst]}
        LSP.append(gloss_dict)

def main():
    args = parse_arguments()

    args.inputVideoPath = normpath(args.inputVideoPath)
    dict_output = normpath(join(args.dict_output, "dict.json"))

    df_video_paths = uv.get_list_data(args.inputVideoPath, ['mp4', 'mov'])

    if args.label_method == 'csv':
        label_dict = read_label_csv(join(args.dict_output, args.csv_name))

        if args.dataset == 'AUTSL':
            train_data = pd.read_csv(join(args.dict_output, 'train_labels.csv'), header=None)
            val_data = pd.read_csv(join(args.dict_output, 'validation_labels.csv'), header=None)
            test_data = pd.read_csv(join(args.dict_output, 'test_labels.csv'), header=None)
            data = pd.concat([train_data, val_data, test_data])

        elif args.dataset == 'LSA64':
            label_dict = read_label_csv(join(args.dict_output, args.csv_name))

    LSP = []

    for num, video_path in tqdm(enumerate(df_video_paths['path']), desc="Processing"):
        process_video(args, label_dict, LSP, normpath(video_path), num)

    df = pd.DataFrame(LSP)
    df.to_json(dict_output, orient='index', indent=2)

if __name__ == "__main__":
    main()
