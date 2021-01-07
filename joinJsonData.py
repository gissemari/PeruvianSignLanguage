# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:44:29 2021

@author: Joe
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

    # ######### JOIN SECTION ##########

    for file in framesToJoin:

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
        
        #print(len(opFace), len(mpFace['x']))
        #print(len(opLeftHand), len(mpHand1['x']))
        #print(len(opRightHand), len(mpHand2['x']))
        print(len(opPose), len(mpPose['x']))
        
        
        