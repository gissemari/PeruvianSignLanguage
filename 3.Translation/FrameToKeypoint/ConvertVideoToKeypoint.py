# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:46:15 2020

@author: Joe

The structure of this three used model are very similar
So I decide to separe their similar parts by section defined by

#########################
# SOMETHING
##############

and

# ###### SOMETHING SECTION #######


"""

import argparse
import cv2
import os
import pandas as pd
import mediapipe as mp
import time
import numpy as np
import pickle as pkl
#########################
# UTIL METODS
##############


def createFolder(path):
    if not os.path.isdir(path):
        print("Directory %s has successfully created" % path)
        os.mkdir(path)


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

# File paths
parser.add_argument('--img_input', type=str,
                    default="./Data/Videos/OnlySquare/frames/",
                    help='relative path of images input.' +
                    ' Default: ./Data/Videos/OnlySquare/frames/')

parser.add_argument('--img_output', type=str, default="/home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos_Keypoints/Segmented_gestures",
                    help='relative path of images output with landmarks.' +
                    ' Default: ./imgOut/mediapipe/')

#parser.add_argument('--pkl_output', type=str, default="./pklOut/", help='relative path of scv output set of landmarks.' + ' Default: ./pklOut/')

parser.add_argument('--json_output', type=str, default="./jsonOut/mediapipe/",
                    help='relative path of scv output set of landmarks.' +
                    ' Default: ./jsonOut/mediapipe/')

parser.add_argument('--pkl_output', type=str, default="/home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Keypoints/Segmented_gestures",
                    help='relative path of csv output set of landmarks.' +
                    ' Default: ./jsonOut/mediapipe/')
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
##############
if(args.face_mesh):
    mp_face_mesh = mp.solutions.face_mesh
if(args.hands):
    mp_hands = mp.solutions.hands
if(args.pose):
    mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
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

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


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
if os.path.isdir(args.img_input):
    folder_list = os.listdir(args.img_input)
    print("Is Directorys")
else:
    folder_list = [args.img_input]

for videoFile in folder_list:

    # Loading
    loading = "Loading:[          ]"
    timer = -1

    list_seq = []


    # Create Folders
    '''
    createFolder(args.pkl_output + folder)
    createFolder(args.pkl_output + folder + "/face")
    createFolder(args.pkl_output + folder + "/hands")
    createFolder(args.pkl_output + folder + "/pose")

    createFolder(args.json_output + folder)
    createFolder(args.json_output + folder + "/face")
    createFolder(args.json_output + folder + "/hands")
    createFolder(args.json_output + folder + "/pose")

    createFolder(args.img_output + folder)
    '''

    # dictionary counter for models results.
    #
    # Face Mesh  -> F
    # Hands      -> H  +  (number of hands detected)
    # Pose       -> P

    counter = dict({('100', '_F'): 0,
                    ('010', '_H1'): 0,
                    ('020', '_H2'): 0,
                    ('001', '_P'): 0,
                    ('110', '_FH1'): 0,
                    ('120', '_FH2'): 0,
                    ('101', '_FP'): 0,
                    ('011', '_H1P'): 0,
                    ('021', '_H2P'): 0,
                    ('111', '_FH1P'): 0,
                    ('121', '_FH2P'): 0,
                    ('000', '_'): 0,  # No model used
                    })

    videoFileName = args.img_input+'/'+videoFile
    videoFolderName = args.img_input.split('/')[-1]
    pcklFileName = args.pkl_output+  '/'+videoFolderName +'/'+videoFile[:-4]+'.pkl'
    videoKeypointName = args.img_output+  '/'+videoFolderName +'/'+videoFile[:-4]+'.png'
    os.mkdir(args.img_output+  '/'+videoFolderName +'/'+videoFile[:-4])

    print("\nReading %s frames\n" % videoFileName)




    # Create a VideoCapture object
    cap = cv2.VideoCapture(videoFileName)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Unable to read camera feed", videoFileName)


    idx =0
    ret, frame = cap.read()
    # While a frame was read
    while ret == True:


        # Convert the BGR image to RGB before processing.
        imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        if(args.face_mesh):
            faceResults = face_mesh.process(imageBGR)

        if(args.hands):
            handsResults = hands.process(imageBGR)

        if(args.pose):
            poseResults = pose.process(imageBGR)

        if(True):
            holisResults = holistic.process(imageBGR)

        # ###### CHECK MODELS RESULTS SECTION #######

        # nameRes will be use to help us recognise which model have results
        # for each model result, a letter that represent that model
        # will be added
        #
        # Face Mesh  -> F
        # Hands      -> H  +  number of hands detected
        # Pose       -> P

        nameRes = '_'
        dictIndex = ''

        #print(len(holisResults.pose_landmarks.landmark), len(holisResults.left_hand_landmarks.landmark)
        #        , len(holisResults.right_hand_landmarks.landmark), len(holisResults.face_landmarks.landmark))


        # temporal variables
        list_X = []
        list_Y = []
        list_Z = []


        # Pose_landmark might already be enough
        for data_point in holisResults.pose_landmarks.landmark:
            list_X.append(data_point.x)
            list_Y.append(data_point.y)

        '''
        for data_point in holisResults.left_hand_landmarks.landmark:
            list_X.append(data_point.landmark.x)
            list_Y.append(data_point.landmark.y)

        for data_point in holisResults.right_hand_landmarks.landmark:
            list_X.append(data_point.landmark.x)
            list_Y.append(data_point.landmark.y)


        for data_point in holisResults.face_landmarks.landmark:
            list_X.append(data_point.landmark.x)
            list_Y.append(data_point.landmark.y)
        '''

        list_seq.append([list_X,list_Y])


        # ###### IMAGE - LANDMARK ANOTATION SECTION #######

        # Draw annotations on the image
        annotated_image = frame.copy()
        annotated_image.flags.writeable = True

        # FACE MESH landmarks
        if(args.face_mesh and args.pkl_output and faceResults.multi_face_landmarks):

            # Add landmarks into the image
            mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=faceResults.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

        # HANDS landmarks
        if(args.hands and args.pkl_output and handsResults.multi_hand_landmarks):

            # In case the model detect two hands
            for hand_landmarks in handsResults.multi_hand_landmarks:
                # Add landmarks into the image
                mp_drawing.draw_landmarks(
                                image=annotated_image,
                                landmark_list=hand_landmarks,
                                connections=mp_hands.HAND_CONNECTIONS)

        # POSE landmarks
        if(args.pose and args.img_output and poseResults.pose_landmarks):

            # Add landmarks into the image
            mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=poseResults.pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS)

        #mp_drawing.draw_landmarks(annotated_image, holisResults.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        #mp_drawing.draw_landmarks(annotated_image, holisResults.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(annotated_image, holisResults.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(annotated_image, holisResults.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        print(videoKeypointName)

        cv2.imwrite("%s.png" % (args.img_output+  '/'+videoFolderName +'/'+videoFile[:-4]+'/'+str(idx)), annotated_image)

        # Print time for each image
        if(args.verbose):
            print(file, time.time()-start_time, " seconds")


        ret, frame = cap.read()
        idx+=1


    new3D = np.asarray(list_seq).reshape((-1,33*2))
    print(new3D.shape)
    with open(pcklFileName, 'wb') as pickle_file:
        pkl.dump(new3D, pickle_file)

    #########################
    # PRINT FOLDER SUMMARY
    ##############

print(loading+" Complete!")
print("\nSummary:  (total: %d)" % len(folder_list))
for index, name in counter:
    print(counter[(index, name)], " <-- ", name)
print()

#########################
# CLOSE MODELS
##############

if(args.face_mesh):
    face_mesh.close()
if(args.hands):
    hands.close()
if(args.pose):
    pose.close()
