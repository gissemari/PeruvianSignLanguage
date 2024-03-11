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

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv


#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description='Mediapipe models ' +
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

parser.add_argument('--pkl_output', type=str, default="./Data/Keypoints/pkl/Segmented_gestures",
                    help='relative path of csv output set of landmarks.' + ' Default: ./jsonOut/mediapipe/')
# Add Line feature
parser.add_argument('--withLineFeature', action="store_true",
                    help='To have dataset x in 3 dimentions')


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

if args.withLineFeature:
    print("   + with Line Feature")
    holistic = mp_holistic.Holistic(upper_body_only=True,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
else:
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
#########################
# UTILS
##############

# Line feature coneccions
LineFeatureConect = [[11, 12], [12, 14], [14, 16], [16, 18],
                     [16, 20], [16, 22], [11, 13], [13, 15],
                     [15, 17], [15, 19], [15, 21],
                     [0, 8], [0, 7]]

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

print(folder_list)
# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolderName in folder_list:

    videoFolderPath = args.inputPath+videoFolderName
    uv.createFolder(args.pkl_output+'/'+videoFolderName)
    uv.createFolder(args.img_output+'/'+videoFolderName)
    uv.createFolder(args.json_output+'/'+videoFolderName)

    videoFolderList = [file for file in os.listdir(videoFolderPath)]

    for videoFile in videoFolderList:
        list_seq = []

        videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

        pcklFileName = args.pkl_output+  '/'+ videoFolderName +'/'+videoFile[:-4]+'.pkl'
        # Creating folder for each gesture in img:
        imgFolder = args.img_output+  '/'+ videoFolderName +'/'+videoFile[:-4]
        uv.createFolder(imgFolder)
        jsonName = args.json_output+  '/'+ videoFolderName +'/'+videoFile[:-4]+'.json'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderPath+'/'+videoFile)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoSegFolderName)

        idx = 0
        ret, frame = cap.read()
        # While a frame was read
        while ret is True:
            # temporal variables
            list_X = []
            list_Y = []
            # list_Z = []

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ###### IMAGE - LANDMARK ANOTATION SECTION #######

            # Draw annotations on the image
            annotated_image = frame.copy()
            annotated_image.flags.writeable = True

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

            if(args.withLineFeature):
                for conections in LineFeatureConect:
                    p1 = holisResults.pose_landmarks.landmark[conections[0]]
                    p2 = holisResults.pose_landmarks.landmark[conections[1]]

                    lineX = p2.x - p1.x
                    lineY = p2.y - p1.y

                    list_X.append(lineX)
                    list_Y.append(lineY)

            if(args.withLineFeature):
                mp_drawing.draw_landmarks(annotated_image,
                                          holisResults.pose_landmarks,
                                          mp_holistic.UPPER_BODY_POSE_CONNECTIONS)
            else:
                mp_drawing.draw_landmarks(annotated_image,
                                          holisResults.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(annotated_image,
                                          holisResults.left_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(annotated_image,
                                          holisResults.right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(annotated_image,
                                          holisResults.face_landmarks,
                                          mp_holistic.FACE_CONNECTIONS)
            list_seq.append([list_X, list_Y])

            cv2.imwrite("%s.png" % (imgFolder+'/'+str(idx)), annotated_image)

            # Print time for each image
            # if(args.verbose):
            #    print(file, time.time()-start_time, " seconds")

            ret, frame = cap.read()
            idx += 1

        new3D = []

        if args.withLineFeature:
            # 25 (points) + 13(lines) * 2 (x and y axes)
            new3D = np.asarray(list_seq).reshape((-1, (25+13)*2))
        else:
            # 33 (pose points)
            # 21 (left hand points)
            # 21 (right hand points)
            # 468 (face mesh points)
            # * 2 (x and y axes)
            new3D = np.asarray(list_seq).reshape((-1, (33+21+21+468)*2))

        print(videoFolderName, videoFile, new3D.shape)

        # Save JSON
        df = pd.DataFrame({'seq': list_seq})
        df.to_json(jsonName)

        # Save Pickle
        print(pcklFileName)
        with open(pcklFileName, 'wb') as pickle_file:
            pkl.dump(new3D, pickle_file)


#########################
# CLOSE MODELS
##############
holistic.close()
