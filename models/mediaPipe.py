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

####### SOMETHING SECTION ####### 


"""
import argparse
import cv2
import pandas as pd
import mediapipe as mp
import time
#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description= 'Mediapipe models ' + 
                                              '(FaceMesh, Hands, Pose)')

# Models
parser.add_argument('--face_mesh', action="store_true",
                    help='Use face mesh model')
parser.add_argument('--hands', action="store_true", help='Use hands model')
parser.add_argument('--pose', action="store_true", help='Use pose model')

# File paths
parser.add_argument('--img_input', type=str, default="./imgIn/",
                    help='relative path of images input.'+
                    ' Default: ./imgIn/')
parser.add_argument('--img_output', type=str, default="./imgOut/",
                    help='relative path of images output with landmarks.'+
                    ' Default: ./imgOut/')
parser.add_argument('--pkl_output', type=str, default="./pklOut/",
                    help='relative path of scv output set of landmarks.'+
                    ' Default: ./pklOut/')
# verbose
parser.add_argument("--verbose", type=int, help="Verbosity")

args = parser.parse_args()

#########################
#########################
# MODELS(Mediapipe) - Notice that this given orden is important
#                     to manage file name results.
#                     (check "dictionary counter" in Utils)
#  1-FaceMesh
#  2-Hands
#  3-Pose
##############
if(args.face_mesh): mp_face_mesh = mp.solutions.face_mesh
if(args.hands): mp_hands = mp.solutions.hands
if(args.pose): mp_pose = mp.solutions.pose

#########################
#########################
# MODELS PARAMETERS
##############

# FACE MESH parameters.
if(args.face_mesh): face_mesh = mp_face_mesh.FaceMesh(
                                        static_image_mode=True,
                                        max_num_faces=1,
                                        min_detection_confidence=0.5)

# HANDS parameters.
if(args.hands): hands = mp_hands.Hands(
                        static_image_mode=True,
                        max_num_hands=2,
                        min_detection_confidence=0.7)

# POSE parameters.
if(args.pose): pose = mp_pose.Pose(
                        static_image_mode=True, 
                        min_detection_confidence=0.5)

#########################
#########################
# UTILS
##############

# Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Files
file_list = ["example_1.jpg","example_2.jpg"] #TODO make it usable with args.img_input

# dictionary counter for models results.
#
# Face Mesh  -> F
# Hands      -> H  +  (number of hands detected)
# Pose       -> P
counter = dict(
            # for 3 used models
            FH1P = 0,
            FH2P = 0,
            
            # for 2 used models
            FP = 0,
            FH1 = 0,
            FH2 = 0,
            HP = 0,
            
            # for 1 used model
            P = 0,
            F = 0,
            H1 = 0,
            H2 = 0)

#########################
#########################
# FILE LIST LOOP
##############
for idx, file in enumerate(file_list):
    
    start_time = time.time()
    
    ####### READ IMAGE SECTION ####### 
    image = cv2.imread(file)
    
    ####### PROCESS MODELS SECTION #######
    
    # Convert the BGR image to RGB before processing.
    imageBGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process
    if(args.face_mesh): faceResults = face_mesh.process(imageBGR)
    if(args.hands):     handsResults = hands.process(imageBGR)
    if(args.pose):      poseResults = pose.process(imageBGR)
    
    ####### CHECK MODELS RESULTS SECTION #######
    
    # nameResult will be use to help us recognise which model have results
    # for each model result, a letter that represent that model will be added
    #
    # Face Mesh  -> F
    # Hands      -> H  +  number of hands detected
    # Pose       -> P
    nameResult = ''
    
    # FACE results
    if(args.face_mesh and faceResults.multi_face_landmarks):
        nameResult += 'F'
    
    # HANDS results
    if(args.hands and handsResults.multi_hand_landmarks):
        nameResult += 'H' + str(len(handsResults.multi_hand_landmarks))
    
    # POSE results
    if(args.pose and poseResults.pose_landmarks):
        nameResult += 'P'
    
    ####### PICKLE - LANDMARK ANOTATION SECTION #######
    

    # temporal variables
    list_X = []
    list_Y = []
    list_Z = []
       
    # Face Mesh landmarks
    if(args.face_mesh and faceResults.multi_face_landmarks):

        for data_point in faceResults.multi_face_landmarks[0].landmark:
            
            list_X.append(data_point.x)
            list_Y.append(data_point.y)
            list_Z.append(data_point.z)
    
        df = pd.DataFrame({
                    'x':list_X,
                    'y':list_Y,
                    'z':list_Z})
        df.to_pickle(args.pkl_output + 'face_annotated_pickle_'+ 
                     str(idx) + '_' + nameResult + '.pkl')
    
    
    list_X.clear()
    list_Y.clear()
    list_Z.clear()
    # Hands landmarks
    if(args.hands and handsResults.multi_hand_landmarks):
        #For each hand
        for hand_landmarks in handsResults.multi_hand_landmarks:
            
            
            for data_point in hand_landmarks.landmark:
                list_X.append(data_point.x)
                list_Y.append(data_point.y)
                list_Z.append(data_point.z)            
            
            #To separate both hands landmarks (both are saved in the same pkl file)
            if(len(handsResults.multi_hand_landmarks)>1): #TODO check another better way to improve this structure
                list_X.append(-9999)
                list_Y.append(-9999)
                list_Z.append(-9999)  
                
        df = pd.DataFrame({
                    'x':list_X,
                    'y':list_Y,
                    'z':list_Z})
        df.to_pickle(args.pkl_output +'hands_annotated_pickle_'+
                     str(idx) + '_' + nameResult + '.pkl')
    
    
    list_X.clear()
    list_Y.clear()
    list_Z.clear()
    # Pose landmarks 
    if(args.pose and poseResults.pose_landmarks):
        
        for hand_landmarks in poseResults.pose_landmarks.landmark:
            
            list_X.append(data_point.x)
            list_Y.append(data_point.y)
            list_Z.append(data_point.z)
        
        df = pd.DataFrame({
                    'x':list_X,
                    'y':list_Y,
                    'z':list_Z})
        df.to_pickle(args.pkl_output + 'pose_annotated_pickle_'+
                     str(idx) + '_' + nameResult + '.pkl')        
        
    ####### IMAGE - LANDMARK ANOTATION SECTION #######
    
    # Draw annotations on the image
    annotated_image = image.copy()
    annotated_image.flags.writeable = True
    
    # FACE MESH landmarks
    if(args.face_mesh and args.img_output and faceResults.multi_face_landmarks):
        
        # Add landmarks into the image
        mp_drawing.draw_landmarks(
                    image = annotated_image,
                    landmark_list = faceResults.multi_face_landmarks[0],
                    connections = mp_face_mesh.FACE_CONNECTIONS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)
            
    # HANDS landmarks 
    if(args.hands and args.img_output and handsResults.multi_hand_landmarks):
        
        #In case the model detect two hands
        for hand_landmarks in handsResults.multi_hand_landmarks:

            # Add landmarks into the image
            mp_drawing.draw_landmarks(
                            image = image,
                            landmark_list = hand_landmarks,
                            connections = mp_hands.HAND_CONNECTIONS)
    
    # POSE landmarks  
    if(args.pose and args.img_output and poseResults.pose_landmarks):
        
        # Add landmarks into the image
        mp_drawing.draw_landmarks(
                        image = annotated_image, 
                        landmark_list = poseResults.pose_landmarks,
                        connections = mp_pose.POSE_CONNECTIONS)
    
    cv2.imwrite(args.img_output + 'annotated_image_'+ 
                str(idx) + '_' + nameResult + '.png', annotated_image)
    
    print(file, time.time()-start_time," seconds")

#########################
#########################
# CLOSE MODELS
##############
if(args.face_mesh): face_mesh.close()
if(args.hands):     hands.close()
if(args.pose):      pose.close()

#########################



