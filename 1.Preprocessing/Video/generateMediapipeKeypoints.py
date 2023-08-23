# Standard library imports
import argparse
import warnings
import os
import sys

# Third party imports
import cv2
import pandas as pd
import numpy as np
import h5py

# Local imports
sys.path.append('../../')
import utils.video as uv
import utils.mediapipe_functions as mpf

# Title
parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')

# File paths
parser.add_argument('--inputPath', type=str, default="../../Data/AEC/Videos/SEGMENTED_SIGN/",
                    help='relative path of images input.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')
parser.add_argument('--dict_output', type=str, default="../../Data/AEC/",
                    help='relative path of scv output set of landmarks.' +' Default: ./Data/Dataset/dict/')
parser.add_argument('--keypoints_output', type=str, default="../../Data/AEC/AEC_mediapipe.hdf5",
                    help='relative path of csv output set of landmarks.' + ' Default: ./Data/Dataset/keypoints/')
args = parser.parse_args()

args.inputPath = os.path.normpath(args.inputPath)
args.dict_output = os.path.normpath(os.sep.join([args.dict_output,"dict.json"]))

df_video_paths = uv.get_list_data(args.inputPath, ['mp4', 'mov'])

holistic = mpf.model_init()

h5_file = h5py.File(args.keypoints_output, 'w')
LSP = []

video_errors = []

for _num, videoPath in enumerate(df_video_paths['path']):
    print(videoPath)

    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (cap.isOpened() is False):
        print("Unable to read camera feed", videoPath)
        video_errors.append(videoPath)
        continue

    # Read video and collect results
    ret, frame = cap.read()
    results = []
    while ret is True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            frame_kp = mpf.frame_process(holistic, frame)
            results.append(frame_kp)

        ret, frame = cap.read()

    results = np.array(results)
    # get the file name -> split the ext -> split by "_" -> upper
    label = os.path.splitext(os.path.basename(videoPath))[0].split('_')[0].upper()
    unique_name = os.sep.join(videoPath.split(os.sep)[-2:])
    
    # accumulate data
    grupo_name = f"{_num}"
    h5_file.create_group(grupo_name)
   
    h5_file[grupo_name]['video_name'] = unique_name
    h5_file[grupo_name]['label'] = label
    h5_file[grupo_name]['data'] = results

    print(f"Video processed: {_num}")

    glossInst = {
        "image_dimention": {
            "height": frame_height,
            "witdh": frame_width
        },
        #"keypoints_iD": f"{num}",
        #"image_path": pklImagePath,
        "frame_end": results.shape[0],
        "frame_start": 1,
        "instance_id": _num,
        "signer_id": -1,
        "unique_name": unique_name,
        "source": "LSP",
        "split": "",
        "variation_id": -1,
        "source_video_name": os.sep.join(videoPath.split(os.sep)[-2:-1]),
        "timestep_vide_name": os.path.splitext(os.path.basename(videoPath))[0]
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
    df.to_json(args.dict_output, orient='index', indent=2)

h5_file.close()
mpf.close_model(holistic)