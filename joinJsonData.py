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
import math

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

        rightWrist = []
        leftWrist = []

        #################
        # MEDIAPIPE DATA
        ###########
        jsonMP = pd.read_json(
            "./jsonOut/mediapipe/" + videoFolder + '/' + file).to_dict()

        mpFace = jsonMP.get('face')
        mpHand1 = jsonMP.get('hand_1')
        mpHand2 = jsonMP.get('hand_2')
        mpPose = jsonMP.get('pose')

        mpLeftHand = {'x': [], 'y': []}
        mpRightHand = {'x': [], 'y': []}

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

        # if mediapipe have results
        if (len(mpPose['x']) != 0):

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

        # This section will be use to know
        # which side the hand belongs to
        # (right or left)

        if(len(mpHand1['x']) != 0):

            if(len(rightWrist) != 0 and len(leftWrist) != 0):

                right1X = math.pow(mpHand1['x'][0] - rightWrist[0], 2)
                right1Y = math.pow(mpHand1['y'][0] - rightWrist[1], 2)
                distanceRight1 = right1X + right1Y

                left1X = math.pow(mpHand1['x'][0] - leftWrist[0], 2)
                left1Y = math.pow(mpHand1['y'][0] - leftWrist[1], 2)
                distanceLeft2 = left1X + left1Y

                if(distanceRight1 < distanceLeft2):

                    mpRightHand['x'] = mpHand1['x']
                    mpRightHand['y'] = mpHand1['y']

                else:

                    mpLeftHand['x'] = mpHand1['x']
                    mpLeftHand['y'] = mpHand1['y']

        if(len(mpHand2['x']) != 0):

            if(len(rightWrist) != 0 and len(leftWrist) != 0):

                right2X = math.pow(mpHand2['x'][0] - rightWrist[0], 2)
                right2Y = math.pow(mpHand2['y'][0] - rightWrist[1], 2)
                distanceRight2 = right2X + right2Y

                left2X = math.pow(mpHand2['x'][0] - leftWrist[0], 2)
                left2Y = math.pow(mpHand2['y'][0] - leftWrist[1], 2)
                distanceLeft2 = left2X + left2Y

                if(distanceRight2 < distanceLeft2):

                    mpRightHand['x'] = mpHand2['x']
                    mpRightHand['y'] = mpHand2['y']

                else:

                    mpLeftHand['x'] = mpHand2['x']
                    mpLeftHand['y'] = mpHand2['y']

        #################
        # LEFT HAND
        ###########
        # This section will be use
        # to join Left hand

        joinedLeftHand = {}

        # if mediapipe have results
        if(len(opLeftHand) != 0):

            newLeftHandX = []
            newLeftHandY = []

            for index, val in enumerate(opLeftHand):
                if(index % 3 == 2):
                    newLeftHandX.append(opLeftHand[index-2])
                    newLeftHandY.append(opLeftHand[index-1])

            joinedLeftHand['x'] = newLeftHandX
            joinedLeftHand['y'] = newLeftHandY

        elif(len(mpLeftHand['x']) != 0):

            joinedLeftHand['x'] = mpLeftHand['x']
            joinedLeftHand['y'] = mpLeftHand['y']

        #################
        # RIGHT HAND
        ###########
        # This section will be use
        # to join Right hand

        joinedRightHand = {}

        # if mediapipe have results
        if(len(opRightHand) != 0):

            newRightHandX = []
            newRightHandY = []

            for index, val in enumerate(opRightHand):
                if(index % 3 == 2):
                    newRightHandX.append(opRightHand[index-2])
                    newRightHandY.append(opRightHand[index-1])

            joinedRightHand['x'] = newRightHandX
            joinedRightHand['y'] = newRightHandY

        elif(len(mpRightHand['x']) != 0):

            joinedRightHand['x'] = mpRightHand['x']
            joinedRightHand['y'] = mpRightHand['y']

        #################
        # FACE
        ###########

        #print(len(opFace), len(mpFace['x']))
        #print(len(opLeftHand), len(mpHand1['x']))
        #print(len(opRightHand), len(mpHand2['x']))
        #print(len(opPose), len(mpPose['x']))
        
        
        