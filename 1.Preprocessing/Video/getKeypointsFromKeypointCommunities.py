# Standard library imports
import argparse
import os
import sys

# Third party imports
import cv2
import openpifpaf

import numpy as np
import pandas as pd
import pickle as pkl
import torch
# Local imports
sys.path.append(os.getcwd())
import utils.video as uv

inputPath = "./Data/AEC/Videos/SEGMENTED_SIGN/"
####
# Check if route is a folder. If so, create results for each of the videos
# If no, create results for only the video passed
#
###

# Folder list of videos's frames
if os.path.isdir(os.getcwd()+'/'+ inputPath) or os.path.isdir(inputPath):
    folder_list = [file for file in (os.listdir(os.getcwd()+'/'+inputPath))
                   if (os.path.isdir(os.getcwd()+'/'+inputPath+file) or os.path.isdir(inputPath+file))]
    print("InputPath is Directory\n")
else:
    folder_list = [inputPath]

print("Folder List:\n")
print(folder_list)

IdCount = 1
LSP = []
video_errors = []


model, _ = openpifpaf.network.factory(checkpoint='./1.Preprocessing/Video/sk30_wholebody.pkl')
model = model.to(args.device)


print(model)

#dictPath = args.dict_output+'/'+"dict"+'.json'

# Iterate over the folders of each video in Video/Segmented_SIGN
for videoFolderName in folder_list:

    videoFolderPath = inputPath

    videoFolderList = [file for file in os.listdir(os.sep.join([os.getcwd(),videoFolderPath + videoFolderName]))]

    cropVideoPath = videoFolderPath + videoFolderName
    #uv.createFolder(cropVideoPath, createFullPath=True)
    #uv.createFolder(args.keypoints_output+'/'+videoFolderName, createFullPath=True)

    for videoFile in videoFolderList:

        word = videoFile.split("_")[0]
        #if word not in ["G-R", "bien", "comer", "cuánto", "dos", "porcentaje", "proteína", "sí", "tú", "yo"]:
        #    continue

        keypointsDict = []

        videoSegFolderName = videoFolderPath+videoFolderName+'/'+videoFile
        videoSegFolderName = videoSegFolderName.split('/')
        videoSegFolderName = os.sep.join([*videoSegFolderName])

        #pklInitPath = os.sep.join([*args.keypoints_output.split('/')])

        #pklKeypointsCompletePath = os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl'])

        #pklKeypointsPath = os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl'])
        print(videoSegFolderName)

        # Create a VideoCapture object
        cap = cv2.VideoCapture(os.getcwd()+'/'+videoSegFolderName)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", os.getcwd()+videoSegFolderName)
            video_errors.append(videoFolderPath+'/'+videoFile)
            continue

        uniqueVideoName = word + '_' + str(IdCount)
        #video = cv2.VideoWriter(cropVideoPath + word + '_' + str(IdCount)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))
        #video = cv2.VideoWriter(cropVideoPath+'/' + uniqueVideoName +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))

        idx = 0

        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Init video size:",w_frame, h_frame)
        ret, frame = cap.read()

        '''
        # While a frame was read
        while ret is True:

            idx += 1  # Frame count starts at 1

            # temporal variables
            kpDict = {}

            # Convert the BGR image to RGB before processing.
            imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, _ = imageBGR.shape

            holisResults = 
            # POSE

            kpDict["pose"]={}
            if holisResults.pose_landmarks:

                kpDict["pose"]["x"] = [point.x for point in holisResults.pose_landmarks.landmark]
                kpDict["pose"]["y"] = [point.y for point in holisResults.pose_landmarks.landmark]

            else:
                kpDict["pose"]["x"] = [1.0 for point in range(0, 33)]
                kpDict["pose"]["y"] = [1.0 for point in range(0, 33)]

            # HANDS

            # Left hand
            kpDict["left_hand"]={}
            if(holisResults.left_hand_landmarks):

                kpDict["left_hand"]["x"] = [point.x for point in holisResults.left_hand_landmarks.landmark]
                kpDict["left_hand"]["y"] = [point.y for point in holisResults.left_hand_landmarks.landmark]

            else:
                kpDict["left_hand"]["x"] = [1.0 for point in range(0, 21)]
                kpDict["left_hand"]["y"] = [1.0 for point in range(0, 21)]

            # Right hand
            kpDict["right_hand"]={}
            if(holisResults.right_hand_landmarks):

                kpDict["right_hand"]["x"] = [point.x for point in holisResults.right_hand_landmarks.landmark]
                kpDict["right_hand"]["y"] = [point.y for point in holisResults.right_hand_landmarks.landmark]

            else:
                kpDict["right_hand"]["x"] = [1.0 for point in range(0, 21)]
                kpDict["right_hand"]["y"] = [1.0 for point in range(0, 21)]

            # Face mesh

            kpDict["face"]={}

            if(holisResults.face_landmarks):

                kpDict["face"]["x"] = [point.x for point in holisResults.face_landmarks.landmark]
                kpDict["face"]["y"] = [point.y for point in holisResults.face_landmarks.landmark]
            else:
                kpDict["face"]["x"] = [1.0 for point in range(0, 468)]
                kpDict["face"]["y"] = [1.0 for point in range(0, 468)]

            keypointsDict.append(kpDict)
            #video.write(imageBGR)
            # Next frame
            ret, frame = cap.read()
        #video.release()

        height, width, channels = imageBGR.shape
        print("N° frames:",idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        glossInst = {
                "image_dimention": {
                    "height": height,
                    "witdh": width
                },
                "keypoints_path": pklKeypointsPath,
                #"image_path": pklImagePath,
                "frame_end": idx,
                "frame_start": 1,
                "instance_id": IdCount,
                "signer_id": -1,
                "unique_name": uniqueVideoName,
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
        # 87 (face mesh points)
        # * 2 (x and y axes)
        #
        # order = ‘F’ means to read / write the elements using Fortran-like index order
        # So the result of the reshape will change 
        # from => (x0,x1,x2,...,Xn, y0,y1,y2,...,Yn)
        # to => (x0,y0,x1,y1,.., Xn,Yn)
        # That means that features will have the following order: 
        # [Pose, Hand left, Hand right, face] with its corresponding size
        # keypointsData = np.asarray(list_seq).reshape(-1, (33+21+21+0)*2, order="F")

        print(videoFolderPath, videoFile, "\nkeypoints path:", pklKeypointsCompletePath)
        print("Unique name path:",os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl']))

        # Save Pickle

        with open(pklKeypointsCompletePath, 'wb') as pickle_file:
            print()
            pkl.dump(keypointsDict, pickle_file)

        print()
        
        # Save JSON
        df = pd.DataFrame(LSP)
        df.to_json(dictPath, orient='index', indent=2)
        
        # Id of each instance
        IdCount += 1

if not video_errors:
    print("No errors founded in any videos")
else:
    print("\nErrors founded in:\n")
    for error in video_errors:
        print(error)


#########################
# CLOSE MODELS
##############

'''