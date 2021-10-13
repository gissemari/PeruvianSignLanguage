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
parser.add_argument('--face', action="store_true",
                    help='Use holistic model: face')

parser.add_argument('--hands', action="store_true",
                    help='Use holistic model: hands')

parser.add_argument('--pose', action="store_true",
                    help='Use holistic model: pose')

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

parser.add_argument("--timeStepSize", type=int, default=17,
                    help="Number of timestep size you want")

# verbose
parser.add_argument("--verbose", type=int, help="Verbosity")

args = parser.parse_args()

if not (args.image or args.face or args.hands or args.pose):
    print("NO MODEL WAS SELECTED")
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

IdCount = 1
LSP = []
video_errors = []

dictPath = args.dict_output+'/'+"dict"+'.json'

# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolderName in folder_list:
    print()
    videoFolderPath = args.inputPath+videoFolderName

    videoFolderList = [file for file in os.listdir(videoFolderPath)]

    for videoFile in videoFolderList:

        word = videoFile.split('_')[0]

        list_seq = []

        videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

        pklKeypointsPath = args.keypoints_output+str(IdCount)+'.pkl'
        pklImagePath = args.img_output+str(IdCount)+'.pkl'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderPath+'/'+videoFile)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoSegFolderName)
            video_errors.append(videoSegFolderName)
            continue
        
        if args.image:
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
            if(args.pose):
                for posi, data_point in enumerate(holisResults.pose_landmarks.landmark):
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
            
            # HANDS
            if(args.hands):
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
            if args.face:
                
                if(holisResults.face_landmarks):

                    # nose points = [1,5,6,218,438]
                    # mouth points = [0,37,39,40,61,185,267,269,270,291,409, 
                    #                 12,38,41,42,62,183,268,271,272,292,407,
                    #                 15,86,89,96,179,316,319,325,403,
                    #                 17,84,91,146,181,314,321,375,405]
                    # left eyes points = [33,133,157,158,159,160,161,173,246,
                    #                     7,144,145,153,154,155,163]
                    # left eyebrow points = [63,66,70,105,107,
                    #                        46,52,53,55,65]
                    # right eyes points = [263,362,384,385,386,387,388,398,466,
                    #                      249,373,374,380,381,382,390]
                    # right eyebrow points = [293,296,300,334,336,
                    #                         276,282,283,285,295]
                    #  
                    #There are 97 points
                    exclusivePoints = [1,5,6,218,438,
                                      0,37,39,40,61,185,267,269,270,291,409, 
                                      12,38,41,42,62,183,268,271,272,292,407,
                                      15,86,89,96,179,316,319,325,403,
                                      17,84,91,146,181,314,321,375,405,
                                      33,133,157,158,159,160,161,173,246,
                                      7,144,145,153,154,155,163,
                                      63,66,70,105,107,
                                      46,52,53,55,65,
                                      263,362,384,385,386,387,388,398,466,
                                      249,373,374,380,381,382,390,
                                      293,296,300,334,336,
                                      276,282,283,285,295]
                    
                    for posi, data_point in enumerate(holisResults.face_landmarks.landmark):
                        if posi in exclusivePoints:
                            list_X.append(data_point.x)
                            list_Y.append(data_point.y)
                else:
                    for _ in range(0, len(exclusivePoints)):
                        list_X.append(1.0)
                        list_Y.append(1.0)

            if args.image:
                # acumulate image frame as a data
                image_data_acum.append(imageBGR)

            # union of x and y keypoints axes
            list_seq.append([list_X, list_Y])

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
        # 97 (face mesh points)
        # * 2 (x and y axes)
        #
        # order = ‘F’ means to read / write the elements using Fortran-like index order
        # So the result of the reshape will change 
        # from => (x0,x1,x2,...,Xn, y0,y1,y2,...,Yn)
        # to => (x0,y0,x1,y1,.., Xn,Yn)
        # That means that features will have the following order: 
        # [Pose, Hand left, Hand right, face] with its corresponding size
        keypointsData = np.asarray(list_seq).reshape(-1, (33+21+21+97)*2, order="F")

        if args.image:
            imageData = np.asarray(image_data_acum)
            # to change from (T, H, W, C) to (T, C, H, W)
            # T = timesteps
            # C = channels = 3
            # H and W are dimentions of the image
            imageData = np.moveaxis(imageData, -1, 1)
            # image s
            imageData = imageData/255
        
        if(args.timeStepSize > 1):

            if len(keypointsData) == args.timeStepSize:
                continue
            # To complete the number of timesteps if it is less than requiered
            elif len(keypointsData) < args.timeStepSize:
                for _ in range(args.timeStepSize - len(keypointsData)):
                    keypointsData = np.append(keypointsData, [keypointsData[-1]], axis=0)
                    if args.image:
                        imageData = np.append(imageData, [imageData[-1]], axis=0)

            # More than the number of timesteps
            else:
                toSkip = len(keypointsData) - args.timeStepSize
                interval = len(keypointsData) // toSkip

                # Generate an interval of index
                a = [val for val in range(0, len(keypointsData)) if val % interval == 0]
    
                # from the list of index, we erase only the number of index we want to skip
                keypointsData = np.delete(keypointsData, a[-toSkip:], axis=0)
                if args.image:
                    imageData = np.delete(imageData, a[-toSkip:], axis=0)

            LSP[glossPos]["instances"][-1]["frame_end"] = args.timeStepSize
        
        print(videoFolderName, videoFile, "\nkeypoints shape:", keypointsData.shape)
        if args.image:
            print("Image shape:", imageData.shape)
        
        # Save Pickle

        with open(pklKeypointsPath, 'wb') as pickle_file:
            pkl.dump(keypointsData, pickle_file)

        if args.image:
            print(pklImagePath)
            with open(pklImagePath, 'wb') as pickle_file:
                pkl.dump(imageData, pickle_file)
        print()
        
        # Save JSON
        df = pd.DataFrame(LSP)
        df.to_json(dictPath, orient='index', indent=2)
        
        # Id of each instance
        IdCount += 1
        
print("\nErrors founded in:\n")
for error in video_errors:
    print(error)


#########################
# CLOSE MODELS
##############
holistic.close()
