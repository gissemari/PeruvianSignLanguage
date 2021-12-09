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
parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')

parser.add_argument('--image', action="store_true",
                    help='to get image data in pkl')

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

print("\n#####\nHolistic Model\n#####\n")
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
    print("InputPath is Directory\n")
else:
    folder_list = [args.inputPath]

print("Folder List:\n")
print(folder_list)

uv.createFolder(args.img_output)
uv.createFolder(args.dict_output)
uv.createFolder(args.keypoints_output)

IdCount = 1
LSP = []
video_errors = []

dictPath = args.dict_output+'/'+"dict"+'.json'

# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolderName in folder_list:
    print()
    videoFolderPath = args.inputPath + videoFolderName

    videoFolderList = [file for file in os.listdir(videoFolderPath)]

    for videoFile in videoFolderList:

        word = videoFile.split('_')[0]

        keypointsDict = []

        videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

        pklKeypointsPath = args.keypoints_output+str(IdCount)+'.pkl'
        pklImagePath = args.img_output+str(IdCount)+'.pkl'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderPath+'/'+videoFile)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoFolderPath+'/'+videoFile)
            video_errors.append(videoFolderPath+'/'+videoFile)
            continue

        if args.image:
            image_data_acum = []

        idx = 0
        continue
        ret, frame = cap.read()
        # While a frame was read
        while ret is True:

            idx += 1  # Frame count starts at 1

            # temporal variables
            kpDict = {}

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ###### IMAGE - LANDMARK ANOTATION SECTION #######
            # Process

            holisResults = holistic.process(imageBGR)

            # POSE

            kpDict["pose"]={}
 
            if holisResults.pose_landmarks:

                kpDict["pose"]["x"] = [point.x for point in holisResults.pose_landmarks.landmark]
                kpDict["pose"]["y"] = [point.y for point in holisResults.pose_landmarks.landmark]

            else:
                kpDict["pose"]["x"] = [1.0 for point in range(0, 33)]
                kpDict["pose"]["y"] = [1.0 for point in range(0, 33)]
        
            # HANDS

            # Left hand
            kpDict["left_hand"]={}
            if(holisResults.left_hand_landmarks):

                kpDict["left_hand"]["x"] = [point.x for point in holisResults.left_hand_landmarks.landmark]
                kpDict["left_hand"]["y"] = [point.y for point in holisResults.left_hand_landmarks.landmark]

            else:
                kpDict["left_hand"]["x"] = [1.0 for point in range(0, 21)]
                kpDict["left_hand"]["y"] = [1.0 for point in range(0, 21)]

            # Right hand
            kpDict["right_hand"]={}
            if(holisResults.right_hand_landmarks):
                kpDict["right_hand"]["x"] = [point.x for point in holisResults.right_hand_landmarks.landmark]
                kpDict["right_hand"]["y"] = [point.y for point in holisResults.right_hand_landmarks.landmark]

            else:
                kpDict["right_hand"]["x"] = [1.0 for point in range(0, 21)]
                kpDict["right_hand"]["y"] = [1.0 for point in range(0, 21)]

            # Face mesh

            kpDict["face"]={}

            if(holisResults.face_landmarks):
                '''
                nose_points = [1,5,6,218,438]
                mouth_points = [78,191,80,81,82,13,312,311,310,415,308,
                                95,88,178,87,14,317,402,318,324,
                                61,185,40,39,37,0,267,269,270,409,291,
                                146,91,181,84,17,314,405,321,375]
                #mouth_points = [0,37,39,40,61,185,267,269,270,291,409, 
                #                12,38,41,42,62,183,268,271,272,292,407,
                #                15,86,89,96,179,316,319,325,403,
                #                17,84,91,146,181,314,321,375,405]
                left_eyes_points = [33,133,157,158,159,160,161,173,246,
                                    7,144,145,153,154,155,163]
                left_eyebrow_points = [63,66,70,105,107]
                                       #46,52,53,55,65]
                right_eyes_points = [263,362,384,385,386,387,388,398,466,
                                     249,373,374,380,381,382,390]
                right_eyebrow_points = [293,296,300,334,336]
                                        #276,282,283,285,295]
  
                #There are 97 points
                exclusivePoints = nose_points
                exclusivePoints = exclusivePoints + mouth_points
                exclusivePoints = exclusivePoints + left_eyes_points
                exclusivePoints = exclusivePoints + left_eyebrow_points
                exclusivePoints = exclusivePoints + right_eyes_points
                exclusivePoints = exclusivePoints + right_eyebrow_points
                '''

                kpDict["face"]["x"] = [point.x for point in holisResults.face_landmarks.landmark]
                kpDict["face"]["y"] = [point.y for point in holisResults.face_landmarks.landmark]

                '''
                for posi, data_point in enumerate(holisResults.face_landmarks.landmark):
                    if posi in exclusivePoints:
                        list_X.append(data_point.x)
                        list_Y.append(data_point.y)
                '''
            else:
                kpDict["face"]["x"] = [1.0 for point in range(0, 468)]
                kpDict["face"]["y"] = [1.0 for point in range(0, 468)]
        
            keypointsDict.append(kpDict)
            if args.image:
                # acumulate image frame as a data
                image_data_acum.append(imageBGR)

            # Next frame
            ret, frame = cap.read()

        height, width, channels = imageBGR.shape

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
        if glossPos != -1:
            LSP[glossPos]["instances"].append(glossInst)
        else:
            glossDict = {"gloss": str(word),
                         "instances": [glossInst]
                         }
            LSP.append(glossDict)

        keypointsData = []
        imageData = []

        # 33 (pose points)
        # 21 (left hand points)
        # 21 (right hand points)
        # 87 (face mesh points)
        # * 2 (x and y axes)
        #
        # order = ‘F’ means to read / write the elements using Fortran-like index order
        # So the result of the reshape will change 
        # from => (x0,x1,x2,...,Xn, y0,y1,y2,...,Yn)
        # to => (x0,y0,x1,y1,.., Xn,Yn)
        # That means that features will have the following order: 
        # [Pose, Hand left, Hand right, face] with its corresponding size
        # keypointsData = np.asarray(list_seq).reshape(-1, (33+21+21+0)*2, order="F")

        if args.image:
            imageData = np.asarray(image_data_acum)
            # to change from (T, H, W, C) to (T, C, H, W)
            # T = timesteps
            # C = channels = 3
            # H and W are dimentions of the image
            imageData = np.moveaxis(imageData, -1, 1)
            # image s
            imageData = imageData/255

        print(videoFolderPath, videoFile, "\nkeypoints path:", pklKeypointsPath)

        if args.image:
            print("Image shape:", imageData.shape)

        # Save Pickle

        with open(pklKeypointsPath, 'wb') as pickle_file:
            pkl.dump(keypointsDict, pickle_file)

        if args.image:
            print("Image path:",pklImagePath)
            with open(pklImagePath, 'wb') as pickle_file:
                pkl.dump(imageData, pickle_file)
        print()
        
        # Save JSON
        df = pd.DataFrame(LSP)
        df.to_json(dictPath, orient='index', indent=2)
        
        # Id of each instance
        IdCount += 1

if not video_errors:
    print("No errors founded in any videos")
else:
    print("\nErrors founded in:\n")
    for error in video_errors:
        print(error)


#########################
# CLOSE MODELS
##############
holistic.close()
