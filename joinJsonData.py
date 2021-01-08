# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:44:29 2021

@author: Joe

All information about keypoints was gotten from:

# Mediapipe

Pose: https://google.github.io/mediapipe/solutions/pose.html
hands: https://google.github.io/mediapipe/solutions/hands.html
Facemesh: https://github.com/tensorflow/tfjs-models/tree/master/facemesh

# Openpose
https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#json-ui-mapping

"""
import os
import pandas as pd

path = "./jsonOut/joined"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

mediapipe_folder_list = os.listdir("./jsonOut/mediapipe/")
openpose_folder_list = os.listdir("./jsonOut/openpose/")

# to Join folders that are similar in both model results
foldersToJoin = []

for mediapipeFile in mediapipe_folder_list:

    if(mediapipeFile in openpose_folder_list):
        foldersToJoin.append(mediapipeFile)

############################
# JOIN JSON RESULTS
##########
for videoFolder in foldersToJoin:

    # Create folders which will be used to join
    subfolder = path + '/' + videoFolder
    if not os.path.isdir(subfolder):
        print("Directory %s has successfully created" % subfolder)
        os.mkdir(subfolder)

    # MEDIAPIPE
    mp_vid_frames = os.listdir("./jsonOut/mediapipe/"+videoFolder)

    # OPENPOSE
    op_vid_frames = os.listdir("./jsonOut/openpose/"+videoFolder)

    # to Join frames that are similar in both model results
    framesToJoin = []

    for mediapipeFrame in mp_vid_frames:

        if(mediapipeFrame in op_vid_frames):
            framesToJoin.append(mediapipeFrame)

    joinedData = {}

    # ######### JOIN SECTION ##########

    for file in framesToJoin:

        #################
        # UTILS
        ###########

        rightWrist = -1
        leftWrist = -1

        #################
        # MEDIAPIPE DATA
        ###########
        jsonMP = pd.read_json(
            "./jsonOut/mediapipe/" + videoFolder + '/' + file).to_dict()

        mpFace = jsonMP.get('face')
        mpHand1 = jsonMP.get('hand_1')
        mpHand2 = jsonMP.get('hand_2')
        mpPose = jsonMP.get('pose')

        #################
        # OPENPOSE DATA
        ###########
        jsonOP = pd.read_json(
            "./jsonOut/openpose/" + videoFolder + '/' + file).to_dict()

        opFace = []
        opLeftHand = []
        opRightHand = []
        opPose = []

        if len(jsonOP.get('people')) > 0:

            opFace = jsonOP.get('people')[0].get('face_keypoints_2d')
            opLeftHand = jsonOP.get('people')[0].get('hand_left_keypoints_2d')
            opRightHand = jsonOP.get('people')[0].get(
                'hand_right_keypoints_2d')
            opPose = jsonOP.get('people')[0].get('pose_keypoints_2d')

        #################
        # POSE
        ###########

        joinedPose = {}
        # Results in both models
        if (len(mpPose['x']) != 0 and len(opPose) != 0):

            rightWrist = [mpPose['x'][16], mpPose['y'][16]]
            leftWrist = [mpPose['x'][15], mpPose['y'][15]]

            indexOrder = [11, 12, 13, 14, 15, 16]

            joinedPose['x'] = [mpPose['x'][pos]*220 for pos in indexOrder]
            joinedPose['y'] = [mpPose['y'][pos]*220 for pos in indexOrder]

        # if mediapipe have results
        elif (len(mpPose['x']) != 0):

            rightWrist = [mpPose['x'][16], mpPose['y'][16]]
            leftWrist = [mpPose['x'][15], mpPose['y'][15]]

            indexOrder = [11, 12, 13, 14, 15, 16]

            joinedPose['x'] = [mpPose['x'][pos]*220 for pos in indexOrder]
            joinedPose['y'] = [mpPose['y'][pos]*220 for pos in indexOrder]

        # if openpose have results
        elif (len(opPose) != 0):

            newPoseX = []
            newPoseY = []

            for index, val in enumerate(opPose):
                if(index % 3 == 2):
                    newPoseX.append(opPose[index-2])
                    newPoseY.append(opPose[index-1])

            rightWrist = [newPoseX[4], newPoseY[4]]
            leftWrist = [newPoseX[7], newPoseY[7]]

            indexOrder = [5, 2, 6, 3, 7, 4]

            joinedPose['x'] = [newPoseX[pos] for pos in indexOrder]
            joinedPose['y'] = [newPoseY[pos] for pos in indexOrder]

        # No results in both models
        else:
            joinedPose['x'] = []
            joinedPose['y'] = []

        joinedData['pose'] = joinedPose

        #################
        # HANDs
        ###########



        #################
        # FACE
        ###########
        

        

        
        #print(len(opFace), len(mpFace['x']))
        #print(len(opLeftHand), len(mpHand1['x']))
        #print(len(opRightHand), len(mpHand2['x']))
        #print(len(opPose), len(mpPose['x']))
        
        
        