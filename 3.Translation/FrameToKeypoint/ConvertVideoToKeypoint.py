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
parser.add_argument('--face_mesh', action="store_true",
                    help='Use face mesh model')
parser.add_argument('--hands', action="store_true", help='Use hands model')
parser.add_argument('--pose', action="store_true", help='Use pose model')
parser.add_argument('--holistic', action="store_true",
                    help='Use holistic model: face, hands and pose')

# File paths
parser.add_argument('--inputPath', type=str, default="./Data/Videos/Segmented_gestures/",
                    help='relative path of images input.' + ' Default: ./Data/Videos/OnlySquare/frames/')

parser.add_argument('--img_output', type=str, default="./Data/Keypoints/png/Segmented_gestures",
                    help='relative path of images output with landmarks.' + ' Default: ./imgOut/mediapipe/')

parser.add_argument('--json_output', type=str, default="./Data/Keypoints/json/Segmented_gestures",
                    help='relative path of scv output set of landmarks.' +' Default: ./jsonOut/mediapipe/')

parser.add_argument('--pkl_output', type=str, default="./Data/Keypoints/pkl/Segmented_gestures",
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
if(args.face_mesh):
    mp_face_mesh = mp.solutions.face_mesh
if(args.hands):
    mp_hands = mp.solutions.hands
if(args.pose):
    mp_pose = mp.solutions.pose
if(args.holistic):
    mp_holistic = mp.solutions.holistic


#########################
# MODELS PARAMETERS
##############

# FACE MESH parameters.
if(args.face_mesh):
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                      max_num_faces=1,
                                      min_detection_confidence=0.5)

# HANDS parameters.
if(args.hands):
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=2,
                           min_detection_confidence=0.7)

# POSE parameters.
if(args.pose):
    pose = mp_pose.Pose(static_image_mode=True,
                        min_detection_confidence=0.5)

if (args.holistic):
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

print(folder_list)
# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolder in folder_list:
    videoFolderName = args.inputPath+videoFolder
    uv.createFolder(args.pkl_output+'/'+videoFolder)
    uv.createFolder(args.img_output+'/'+videoFolder)
    uv.createFolder(args.json_output+'/'+videoFolder)

    videoFolderList = [file for file in os.listdir(videoFolderName)]

    for videoFile in videoFolderList:
        list_seq = []

        videoSegFolderName = videoFolderName+'/'+videoFile[:-4]
        # videoFolderName = args.inputPath.split('/')[-1]
        pcklFileName = args.pkl_output+  '/'+videoFolder +'/'+videoFile[:-4]+'.pkl'
        # Creating folder for each gesture in img:
        imgFolder = args.img_output+  '/'+videoFolder +'/'+videoFile[:-4]
        uv.createFolder(imgFolder)
        jsonName = args.json_output+  '/'+videoFolder +'/'+videoFile[:-4]+'.json'

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderName+'/'+videoFile)

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
            #list_Z = []

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ###### IMAGE - LANDMARK ANOTATION SECTION #######

            # Draw annotations on the image
            annotated_image = frame.copy()
            annotated_image.flags.writeable = True

            # Process
            if(args.face_mesh):
                faceResults = face_mesh.process(imageBGR)

                for data_point in faceResults.multi_face_landmarks[0].landmark:
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
                    # list_Z.append(data_point.z)

                mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=faceResults.multi_face_landmarks[0],
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)

            if(args.hands):
                handsResults = hands.process(imageBGR)

                # For each hand
                for hand_landmarks in handsResults.multi_hand_landmarks:
                    for data_point in hand_landmarks.landmark:
                        list_X.append(data_point.x)
                        list_Y.append(data_point.y)
                        # list_Z.append(data_point.z)

                for hand_landmarks in handsResults.multi_hand_landmarks:
                    # Add landmarks into the image
                    mp_drawing.draw_landmarks(
                                    image=annotated_image,
                                    landmark_list=hand_landmarks,
                                    connections=mp_hands.HAND_CONNECTIONS)

            if(args.pose):
                poseResults = pose.process(imageBGR)

                for hand_landmarks in poseResults.pose_landmarks.landmark:

                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
                    # list_Z.append(data_point.z)

                # Add landmarks into the image
                mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=poseResults.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS)
            if(args.holistic):
                holisResults = holistic.process(imageBGR)
                # Pose_landmark might already be enough
                # for data_point in faceResults.landmarkS:
                #     list_X.append(data_point.landmark.x)
                #     list_Y.append(data_point.landmark.y)

                # for data_point in holisResults.left_hand_landmarks.landmark:
                #     list_X.append(data_point.landmark.x)
                #     list_Y.append(data_point.landmark.y)

                # for data_point in holisResults.right_hand_landmarks.landmark:
                #     list_X.append(data_point.landmark.x)
                #     list_Y.append(data_point.landmark.y)

                for data_point in holisResults.pose_landmarks.landmark:
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)

                #mp_drawing.draw_landmarks(annotated_image, holisResults.face_landmarks, mp_holistic.FACE_CONNECTIONS)
                #mp_drawing.draw_landmarks(annotated_image, holisResults.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                #mp_drawing.draw_landmarks(annotated_image, holisResults.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(annotated_image, holisResults.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            list_seq.append([list_X, list_Y])

            cv2.imwrite("%s.png" % (imgFolder+'/'+str(idx)), annotated_image)

            # Print time for each image
            # if(args.verbose):
            #    print(file, time.time()-start_time, " seconds")

            ret, frame = cap.read()
            idx += 1

        new3D = np.asarray(list_seq).reshape((-1, 33*2))
        print(videoFolder, videoFile, new3D.shape)

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

if(args.face_mesh):
    face_mesh.close()
if(args.hands):
    hands.close()
if(args.pose):
    pose.close()
