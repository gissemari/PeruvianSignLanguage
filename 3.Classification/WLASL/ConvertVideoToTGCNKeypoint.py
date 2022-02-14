# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:46:15 2020

@author: Joe

Updated on 29 Dic 2020 by Gissella Bejarano

The structure of this three used model are very similar
So I decide to separe their similar parts by section defined by

#########################
# SOMETHING
##############

and

# ###### SOMETHING SECTION #######


"""
# Standard library imports
import argparse
import os
import sys

# Third party imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle as pkl
from math import hypot

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv   # for folder creation


#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description='Mediapipe models to TGCN' +
                                 '(FaceMesh, Hands, Pose)')

# Models
parser.add_argument('--holistic', action="store_true", help='Use holistic model: face, hands and pose')

# File paths
parser.add_argument('--inputPath', type=str, default="./Data/Videos/Segmented_gestures/",
                    help='relative path of images input.' + ' Default: ./Data/Videos/OnlySquare/frames/')

parser.add_argument('--img_output', type=str, default="./Data/Keypoints/png/Segmented_gestures",
                    help='relative path of images output with landmarks.' + ' Default: ./imgOut/mediapipe/')

parser.add_argument('--json_output', type=str, default="./Data/Keypoints/json/Segmented_gestures",
                    help='relative path of scv output set of landmarks.' +' Default: ./jsonOut/mediapipe/')

parser.add_argument('--pkl_output', type=str, default="./Data/Keypoints/pkl_TGCN/Segmented_gestures",
                    help='relative path of csv output set of landmarks.' + ' Default: ./jsonOut/mediapipe/')


# verbose
parser.add_argument("--verbose", type=int, help="Verbosity")

args = parser.parse_args()


#########################
# MODELS(Mediapipe) - Notice that this given orden is important
#                     to manage file name results.
#                     (check counter variable in FOLDER LIST LOOP)
#  1-FaceMesh
#  2-Hands
#  3-Pose
#  4-Holistic
##############
print()
print("model (using):")

print(" - holistic")
mp_holistic = mp.solutions.holistic
print()

#########################
# MODELS PARAMETERS
##############

holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)
#########################
# UTILS
##############

# Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#########################
# FOLDER LIST LOOP
##############

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

print("Total videos to process: ", len(folder_list))

# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolder in folder_list:

    videoFolderName = args.inputPath+videoFolder
    uv.createFolder(args.pkl_output, createFullPath=True)
    uv.createFolder(args.pkl_output+'/'+videoFolder)
    uv.createFolder(args.img_output+'/'+videoFolder)
    uv.createFolder(args.json_output+'/'+videoFolder)

    videoFolderList = [file for file in os.listdir(videoFolderName)]
    print("Video: ", videoFolder)
    for videoFile in videoFolderList:
        list_seq = []

        videoSegFolderName = videoFolderName+'/'+videoFile[:-4]

        pcklFileName = args.pkl_output+'/'+videoFolder +'/'+videoFile[:-4]+'.pkl'
        # Creating folder for each gesture in img:
        imgFolder = args.img_output+'/'+videoFolder +'/'+videoFile[:-4]
        uv.createFolder(imgFolder)
        jsonName = args.json_output+'/'+videoFolder +'/'+videoFile[:-4]+'.json'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderName+'/'+videoFile)
        print("Processing: ", videoFile)
        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoSegFolderName)

        idx = 0
        ret, frame = cap.read()
        # While a frame was read
        while ret is True:

            # temporal variables
            faceData = []
            leftHandData = []
            rightHandData = []
            poseData = []
            holisticData = []

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ###### IMAGE - LANDMARK ANOTATION SECTION #######

            # Draw annotations on the image
            annotated_image = frame.copy()
            annotated_image.flags.writeable = True


            holisResults = holistic.process(imageBGR)

            # POSE

            if holisResults.pose_landmarks.landmark:
                for landmarks in holisResults.pose_landmarks.landmark:

                    poseData.append(landmarks.x * 256.0)
                    poseData.append(landmarks.y * 256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    poseData.append(landmarks.visibility)

            else:
                for hand_landmarks in range(0, 33):

                    poseData.append(256.0)
                    poseData.append(256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    poseData.append(0.0)
        
            # HANDS

            # Left hand

            if(holisResults.left_hand_landmarks):
                for landmarks in holisResults.left_hand_landmarks.landmark:

                    leftHandData.append(landmarks.x * 256.0)
                    leftHandData.append(landmarks.y * 256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    leftHandData.append(0.0)

            else:
                for hand_landmarks in range(0, 21):

                    leftHandData.append(256.0)
                    leftHandData.append(256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    leftHandData.append(0.0)

            # Right hand

            if(holisResults.right_hand_landmarks):

                for landmarks in holisResults.right_hand_landmarks.landmark:

                    rightHandData.append(landmarks.x * 256.0)
                    rightHandData.append(landmarks.y * 256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    rightHandData.append(0.0)

            else:
                for hand_landmarks in range(0, 21):

                    rightHandData.append(256.0)
                    rightHandData.append(256.0)

                    #visibility: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.
                    rightHandData.append(0.0)

            #POSE Corrections
            exclude = list(range(23,33))

            denied = set([9 , 10, 11, 12, 13, 14, 19, 20, 21, 22]) # 23 and 24 are zeros
            toChange = set(range(1,11))

            x = [v for i, v in enumerate(poseData) if i % 3 == 0 and i // 3 not in exclude]
            y = [v for i, v in enumerate(poseData) if i % 3 == 1 and i // 3 not in exclude]
            p = [v for i, v in enumerate(poseData) if i % 3 == 2 and i // 3 not in exclude]

            x = x + [0.0, 0.0]
            y = y + [0.0, 0.0]
            p = p + [0.0, 0.0]

            inter = denied & toChange

            inter = list(inter)
            denied = list(denied)
            toChange = list(toChange)

            denied = [val for val in denied if val not in inter]
            toChange = [val for val in toChange if val not in inter]

            for pos in range(len(toChange)):
                if pos in inter:
                    continue
                tmp = x[toChange[pos]]
                x[toChange[pos]] = x[denied[pos]]
                x[denied[pos]] = tmp

                tmp = y[toChange[pos]]
                y[toChange[pos]] = y[denied[pos]]
                y[denied[pos]] = tmp

                tmp = p[toChange[pos]]
                p[toChange[pos]] = p[denied[pos]]
                p[denied[pos]] = tmp

            poseData = []
            for val in range(len(x)):
                poseData.append(x[val])
                poseData.append(y[val])
                poseData.append(p[val])

            #Hands Corrections
            if(len(rightHandData) == 0):
                rightHandData = [0]*63
            if(len(leftHandData) == 0):
                leftHandData = [0]*63

            if(len(rightHandData) == 126):

                distance_right = hypot(rightHandData[0]-x[16], rightHandData[1],y[16])
                distance_left = hypot(rightHandData[3*21]-x[16], rightHandData[(3*21)+1],y[16])
                if(distance_right < distance_left):
                    leftHandData = rightHandData[63:126]
                    rightHandData = rightHandData[0:63]
                else:
                    leftHandData = rightHandData[0:63]
                    rightHandData = rightHandData[63:126]

            if(len(leftHandData) == 126):

                distance_left = hypot(leftHandData[0]-x[15], leftHandData[1],y[15])
                distance_right = hypot(leftHandData[3*21]-x[15], leftHandData[(3*21)+1],y[15])

                if(distance_left < distance_right):
                    rightHandData = leftHandData[63:126]
                    leftHandData = leftHandData[0:63]
                else:
                    rightHandData = leftHandData[0:63]
                    leftHandData = leftHandData[63:126]
            if(len(leftHandData)!=63 or len(rightHandData)!=63 or len(poseData)!= 75):
                print("ERROR: l-%d  r-%d p-%d" % (len(leftHandData),len(rightHandData), len(poseData)))

            list_seq.append({
                'pose_keypoints_2d':poseData,
                'hand_left_keypoints_2d': leftHandData,
                'hand_right_keypoints_2d': rightHandData,
                'face_keypoints_2d': faceData
                })

            cv2.imwrite("%s.png" % (imgFolder+'/'+str(idx)), annotated_image)

            # Print time for each image
            # if(args.verbose):
            #    print(file, time.time()-start_time, " seconds")

            ret, frame = cap.read()
            idx += 1

        new3D = np.asarray(list_seq)

        #print(videoFolder, videoFile, new3D.shape)

        # Save JSON
        df = pd.DataFrame({'seq': list_seq})
        df.to_json(jsonName)

        # Save Pickle
        #print(pcklFileName)
        with open(pcklFileName, 'wb') as pickle_file:
            pkl.dump(new3D, pickle_file)


#########################
# CLOSE MODELS
##############

if(args.face_mesh):
    face_mesh.close()
if(args.hands):
    hands.close()
if(args.pose):
    pose.close()
