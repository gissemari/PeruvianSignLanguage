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

parser.add_argument('--img_output', type=str, default="./imgOut/mediapipe/",
                    help='relative path of images output with landmarks.' +
                    ' Default: ./imgOut/mediapipe/')

parser.add_argument('--pkl_output', type=str, default="./pklOut/",
                    help='relative path of scv output set of landmarks.' +
                    ' Default: ./pklOut/')

parser.add_argument('--json_output', type=str, default="./jsonOut/mediapipe/",
                    help='relative path of scv output set of landmarks.' +
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

#########################
# UTILS
##############

# Drawing
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Folder list of videos's frames
folder_list = os.listdir(args.img_input)

#########################
# FOLDER LIST LOOP
##############

for folder in folder_list:

    # frames of one video
    file_list = os.listdir(args.img_input + folder)

    # Loading
    loading = "Loading:[          ]"
    timer = -1

    # Create Folders
    createFolder(args.pkl_output + folder)
    createFolder(args.pkl_output + folder + "/face")
    createFolder(args.pkl_output + folder + "/hands")
    createFolder(args.pkl_output + folder + "/pose")

    createFolder(args.json_output + folder)
    createFolder(args.json_output + folder + "/face")
    createFolder(args.json_output + folder + "/hands")
    createFolder(args.json_output + folder + "/pose")

    createFolder(args.img_output + folder)

    # dictionary counter for models result patterns
    #
    # The key is a pattern where each char corresponds to
    # the number of results that have been generated for each model.
    #
    # First element  => Face Mesh  -> F
    # Second element => Hands      -> H  +  (number of hands detected)
    # Third element  => Pose       -> P
    #
    # The value is an array where the first element shows the number of frames
    # that corresponds with each pattern and the second element shows the
    # meaning of its key pattern

    counter = dict({'100': [0, 'F'],
                    '010': [0, 'H1'],
                    '020': [0, 'H2'],
                    '001': [0, 'P'],
                    '110': [0, 'FH1'],
                    '120': [0, 'FH2'],
                    '101': [0, 'FP'],
                    '011': [0, 'H1P'],
                    '021': [0, 'H2P'],
                    '111': [0, 'FH1P'],
                    '121': [0, 'FH2P'],
                    '000': [0, '_'],  # No model results
                    })

    print("\nReading %s frames" % folder)

    fileNameList = []
    modelsDetectedList = []

    # For each image in the folder
    for idx, file in enumerate(file_list):

        start_time = time.time()

        # Loading Bar
        part = int(idx*10/len(file_list))
        if(timer != part):
            print(loading, "%d/%d" % (idx, len(file_list)))
            for n in range(part - timer):
                loading = loading.replace(" ", "~", 1)
            timer = part

        file_ext = os.path.splitext(file)[1]
        file_name = os.path.splitext(file)[0]

        # ###### READ IMAGE SECTION #######
        image = cv2.imread(args.img_input + folder + '/' + file)

        # ###### PROCESS MODELS SECTION #######

        # Convert the BGR image to RGB before processing.
        imageBGR = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process
        if(args.face_mesh):
            faceResults = face_mesh.process(imageBGR)

        if(args.hands):
            handsResults = hands.process(imageBGR)

        if(args.pose):
            poseResults = pose.process(imageBGR)

        # ###### CHECK MODELS RESULTS SECTION #######

        # modelsDetected will be use to help us recognise which model have
        # results for each model result, a letter that represent that model
        # will be added
        #
        # First element  => Face Mesh  -> F
        # Second element => Hands      -> H  +  (number of hands detected)
        # Third element  => Pose       -> P

        modelsDetected = [0, 0, 0]
        dictIndex = ''

        # FACE results
        if(args.face_mesh and faceResults.multi_face_landmarks):
            modelsDetected[0] = 1

        # HANDS results
        if(args.hands and handsResults.multi_hand_landmarks):
            modelsDetected[1] = len(handsResults.multi_hand_landmarks)

        # POSE results
        if(args.pose and poseResults.pose_landmarks):
            modelsDetected[2] = 1

        # count results by nameResult
        dictIndex = ''.join(str(e) for e in modelsDetected)

        counter[dictIndex][0] += 1

        fileNameList.append(file_name)
        modelsDetectedList.append(modelsDetected)

        # ###### PICKLE - LANDMARK ANOTATION SECTION #######

        # temporal variables
        list_X = []
        list_Y = []
        list_Z = []

        # Face Mesh landmarks
        if(args.face_mesh and faceResults.multi_face_landmarks):

            # X, Y, Z points for each landmark
            for data_point in faceResults.multi_face_landmarks[0].landmark:
                list_X.append(data_point.x)
                list_Y.append(data_point.y)
                list_Z.append(data_point.z)

            df = pd.DataFrame({
                        'x': list_X,
                        'y': list_Y,
                        'z': list_Z})

            df.to_pickle("%sface/%s.pkl" %
                         (args.pkl_output + folder + '/',
                          file_name))

            df.to_json("%sface/%s.json" %
                       (args.json_output + folder + '/',
                        file_name))

        list_X.clear()
        list_Y.clear()
        list_Z.clear()

        # Hands landmarks
        if(args.hands and handsResults.multi_hand_landmarks):

            # For each hand
            for hand_landmarks in handsResults.multi_hand_landmarks:
                for data_point in hand_landmarks.landmark:
                    list_X.append(data_point.x)
                    list_Y.append(data_point.y)
                    list_Z.append(data_point.z)

                # To separate both hands landmarks
                # (both are saved in the same pkl file)

                # TODO check another better way to improve this structure
                if(len(handsResults.multi_hand_landmarks) > 1):
                    list_X.append(-9999)
                    list_Y.append(-9999)
                    list_Z.append(-9999)

            df = pd.DataFrame({
                        'x': list_X,
                        'y': list_Y,
                        'z': list_Z})

            df.to_pickle("%shands/%s.pkl" %
                         (args.pkl_output + folder + '/',
                          file_name))

            df.to_json("%shands/%s.json" %
                       (args.json_output + folder + '/',
                        file_name))
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
                        'x': list_X,
                        'y': list_Y,
                        'z': list_Z})

            df.to_pickle("%spose/%s.pkl" %
                         (args.pkl_output + folder + '/',
                          file_name))

            df.to_json("%spose/%s.json" %
                       (args.json_output + folder + '/',
                        file_name))

        list_X.clear()
        list_Y.clear()
        list_Z.clear()

        # ###### IMAGE - LANDMARK ANOTATION SECTION #######

        # Draw annotations on the image
        annotated_image = image.copy()
        annotated_image.flags.writeable = True

        # FACE MESH landmarks
        if(args.face_mesh and args.img_output
           and faceResults.multi_face_landmarks):

            # Add landmarks into the image
            mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=faceResults.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)

        # HANDS landmarks
        if(args.hands and args.img_output and
           handsResults.multi_hand_landmarks):

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

        cv2.imwrite("%s%s_%d.png" %
                    (args.img_output + folder + '/', file_name, idx),
                    annotated_image)

        # Print time for each image
        if(args.verbose):
            print(file, time.time()-start_time, " seconds")

    #########################
    # PRINT FOLDER SUMMARY
    ##############

    # To save in a csv file information about
    # which model have results in each frame
    index = pd.MultiIndex.from_product([fileNameList], names=['Filename'])
    columns = pd.MultiIndex.from_product([['Face', 'Hands', 'Pose']],
                                         names=['model'])
    df = pd.DataFrame(modelsDetectedList,
                      index=index,
                      columns=columns)

    df.to_csv("%s%s.csv" % (args.json_output + folder + '/', folder))
    df.to_csv("%s%s.csv" % (args.pkl_output + folder + '/', folder))

    # To print Summary data in the terminal
    print(loading+" Complete!")
    print("\nSummary:  (total: %d)" % len(file_list))
    for index in counter:
        print("%4d ---> %4s" % (counter[index][0], counter[index][1]))
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
