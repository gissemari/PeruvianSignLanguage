# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:42:39 2021

@author: Joe
"""

# Standard library imports
import argparse
import os

# Third party imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle as pkl

# Local imports
import utils.video as uv  # for folder creation


#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description='Mediapipe models ' +
                                 '(FaceMesh, Hands, Pose)')

# Models
parser.add_argument('--holistic', action="store_true",
                    help='Use holistic model: face, hands and pose')

# File paths
parser.add_argument('--inputPath', type=str, default="./Data/Videos/Segmented_gestures/",
                    help='relative path of images input.' + ' Default: ./Data/Videos/Segmented_gestures/')

parser.add_argument('--img_output', type=str, default="./Data/Dataset/img/",
                    help='relative path of images output with landmarks.' + ' Default: ./Data/Dataset/img/')

parser.add_argument('--dict_output', type=str, default="./Data/Dataset/dict/",
                    help='relative path of scv output set of landmarks.' +' Default: ./Data/Dataset/dict/')

parser.add_argument('--keypoints_output', type=str, default="./Data/Dataset/keypoints/",
                    help='relative path of csv output set of landmarks.' + ' Default: ./Data/Dataset/keypoints/')

# verbose
parser.add_argument("--verbose", type=int, help="Verbosity")

args = parser.parse_args()


#########################
# MODELS(Mediapipe)
#
# -Holistic
##############

print("Holistic Model")
mp_holistic = mp.solutions.holistic

#########################
# MODELS PARAMETERS
##############

# FACE MESH parameters.
holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

####
# Check if route is a folder. If so, create results for each of the videos
# If no, create results for only the video passed
#
###

# Folder list of videos's frames
if os.path.isdir(args.inputPath):
    folder_list = [file for file in os.listdir(args.inputPath)
                   if os.path.isdir(args.inputPath+file)]
    print("Is Directory")
else:
    folder_list = [args.inputPath]

print(folder_list)

uv.createFolder(args.img_output)
uv.createFolder(args.dict_output)
uv.createFolder(args.keypoints_output)

IdCount = 0
LSP = []

dictPath = args.dict_output+'/'+"dict"+'.json'

# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolderName in folder_list:
    print()
    videoFolderPath = args.inputPath+videoFolderName

    videoFolderList = [file for file in os.listdir(videoFolderPath)]

    for videoFile in videoFolderList:

        word = videoFile.split('_')[0]

        # Id of each instance
        IdCount += 1  # video id starts at 1

        list_seq = []

        videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

        pklKeypointsPath = args.keypoints_output+str(IdCount)+'.pkl'
        pklImagePath = args.img_output+str(IdCount)+'.pkl'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderPath+'/'+videoFile)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoSegFolderName)

        image_data_acum = []

        idx = 0

        ret, frame = cap.read()
        # While a frame was read
        while ret is True:

            idx += 1  # Frame count starts at 1

            # temporal variables
            list_X = []
            list_Y = []

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ###### IMAGE - LANDMARK ANOTATION SECTION #######
            # Process

            holisResults = holistic.process(imageBGR)

            # POSE
            for posi, data_point in enumerate(holisResults.pose_landmarks.landmark):
                list_X.append(data_point.x)
                list_Y.append(data_point.y)

            # Left hand
            if(holisResults.left_hand_landmarks):
                for posi, data_point in enumerate(holisResults.left_hand_landmarks.landmark):
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
            else:
                for _ in range(0, 21): # 21 is the number of points taken in hands model
                    list_X.append(1.0)
                    list_Y.append(1.0)

            # Right hand
            if(holisResults.right_hand_landmarks):
                for posi, data_point in enumerate(holisResults.right_hand_landmarks.landmark):
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
            else:
                for _ in range(0, 21):
                    list_X.append(1.0)
                    list_Y.append(1.0)

            # Face mesh
            if(holisResults.face_landmarks):
                for posi, data_point in enumerate(holisResults.face_landmarks.landmark):
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
            else:
                for _ in range(0, 468):
                    list_X.append(1.0)
                    list_Y.append(1.0)

            # acumulate image frame as a data
            image_data_acum.append(imageBGR)

            # union of x and y keypoints axes
            list_seq.append([list_X, list_Y])

            # Next frame
            ret, frame = cap.read()

        height, width, channels = image_data_acum[0].shape

        glossInst = {
                "image_dimention": {
                    "height": height,
                    "witdh": width
                },
                "keypoints_path": pklKeypointsPath,
                "image_path": pklImagePath,
                "frame_end": idx,
                "frame_start": 1,
                "instance_id": IdCount,
                "signer_id": -1,
                "source": "LSP",
                "split": "",
                "variation_id": -1,
                "source_video_name": videoFolderName,
                "timestep_vide_name": videoFile[:-4]
            }

        # check if there is a gloss asigned with "word"
        glossPos = -1

        for indG, gloss in enumerate(LSP):
            if(gloss["gloss"] == word):
                glossPos = indG

        # in the case word is in the dict
        if glossPos > -1:
            LSP[glossPos]["instances"].append(glossInst)
        else:
            glossDict = {"gloss": str(word),
                         "instances": [glossInst]
                         }
            LSP.append(glossDict)

        new3D = []
        imageData = []

        # 33 (pose points)
        # 21 (left hand points)
        # 21 (right hand points)
        # 468 (face mesh points)
        # * 2 (x and y axes)
        new3D = np.asarray(list_seq).reshape((-1, (33+21+21+468)*2))

        imageData = np.asarray(image_data_acum)
        # to change from (T, H, W, C) to (T, C, H, W)
        # T = timesteps
        # C = channels = 3
        # H and W are dimentions of the image
        imageData = np.moveaxis(imageData, -1, 1)
        # image s
        imageData = imageData/255

        print(videoFolderName, videoFile, "\nkeypoints shape:", new3D.shape,
              "\nImage shape:", imageData.shape)

        # Save Pickle
        print(pklKeypointsPath)
        with open(pklKeypointsPath, 'wb') as pickle_file:
            pkl.dump(new3D, pickle_file)

        print(pklImagePath, "\n")
        with open(pklImagePath, 'wb') as pickle_file:
            pkl.dump(imageData, pickle_file)
        
        # Save JSON
        df = pd.DataFrame(LSP)
        df.to_json(dictPath, orient='index', indent=3)




#########################
# CLOSE MODELS
##############
holistic.close()
