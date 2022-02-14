# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:06:40 2021

@author: Joe
"""
import argparse
import os
import sys
from multiprocessing import Pool
import multiprocessing

# Third party imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle as pkl

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv   # for folder creation and keypoints normalization

print("\n#####\nHolistic Model\n#####\n")
mp_holistic = mp.solutions.holistic

#########################
# MODELS PARAMETERS
##############

# HOLISTIC parameters.
holistic = mp_holistic.Holistic(static_image_mode=True,
                                model_complexity=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


def worker(data):
    
    videoFolderPath = data[0]
    word = data[1]
    IdCount = data[2]
    cropVideoPath = data[3]
    pklKeypointsPath = data[4]
    videoFolderName = data[5]
    videoFile = data[6]
    inputVideo = videoFolderPath+'/'+videoFile
    outputVideo = cropVideoPath + word + '_' + str(IdCount)+'.mp4'

    keypointsDict = []

    # Create a VideoCapture object
    cap = cv2.VideoCapture(inputVideo)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Unable to read camera feed",inputVideo )
        #video_errors.append(inputVideo)
        #continue

    video = cv2.VideoWriter(outputVideo,cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))

    idx = 0

    w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Init video size:", w_frame, h_frame)

    ret, frame = cap.read()

    # While a frame was read
    while ret is True:

        idx += 1  # Frame count starts at 1

        # temporal variables
        kpDict = {}

        # Convert the BGR image to RGB before processing.
        imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = imageBGR.shape

        # ###### IMAGE - LANDMARK ANOTATION SECTION #######
        # Process
        holisResults = holistic.process(imageBGR)
        
        if holisResults.pose_landmarks:
            poseX = [point.x*width for point in holisResults.pose_landmarks.landmark]
            poseY = [point.y*height for point in holisResults.pose_landmarks.landmark]

            sr = np.asarray((poseX[11],poseY[11]))
            sl = np.asarray((poseX[12],poseY[12]))

            sRange = np.linalg.norm(sr - sl)

            mid = (sr+sl)/2

            midY = mid[1]
            prop = 1 - sRange/width

            top = int(midY - sRange*1.2)
            botton = int(midY + sRange*1.2)
            left = int(sl[0] - sRange*prop)
            right = int(sr[0] + sRange*prop)

            bTop, bBot, bLeft, bRight = 0, 0, 0, 0

            if(botton > height):
                bBot = botton - height
                botton = height-1
            if(top < 0):
                bBot = -top
                top = 0
            if(left < 0):
                bLeft = -left
                left = 0
            if(right > width):
                bRight = right - width
                right = width-1

            black = [0,0,0]
            #imageBGR = cv2.copyMakeBorder(imageBGR[top:botton,left:right],bBot,bTop,bLeft,bRight,cv2.BORDER_CONSTANT,value=black)
            imageBGR = cv2.copyMakeBorder(imageBGR[top:botton,left:right],0,0,0,0,cv2.BORDER_CONSTANT,value=black)
        imageBGR = cv2.resize(imageBGR, (220, 220))
        imageBGR = cv2.cvtColor(imageBGR, cv2.COLOR_RGB2BGR)
        
        holisResults = holistic.process(imageBGR)
        
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

        video.write(imageBGR)
        # Next frame
        ret, frame = cap.read()
    video.release()
    
    height, width, channels = imageBGR.shape
    print("N° frames:",idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    glossInst = {
            "image_dimention": {
                "height": height,
                "witdh": width
            },
            "keypoints_path": pklKeypointsPath,
            "frame_end": idx,
            "frame_start": 1,
            "instance_id": IdCount,
            "signer_id": -1,
            "unique_name": word +'_'+ str(IdCount),
            "source": "LSP",
            "split": "",
            "variation_id": -1,
            "source_video_name": videoFolderName,
            "timestep_vide_name": videoFile[:-4]
        }

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


    print(videoFolderPath, videoFile, "\nkeypoints path:", pklKeypointsPath)
    print("Unique name path:", outputVideo[:-4])

    # Save Pickle

    with open(pklKeypointsPath, 'wb') as pickle_file:
        pkl.dump(keypointsDict, pickle_file)

    print()

    print(" Process %s Finished." % inputVideo)
    
    return [glossInst]




def poolHandler(cores, data):
    
    cores = multiprocessing.cpu_count()
    
    p = Pool(cores)
    acum = p.map(worker, data)
    
    LSP = []
    
    for glossInst in acum:
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
    
    df = pd.DataFrame(LSP)
    df.to_json(dictPath, orient='index', indent=2)

if __name__ == '__main__':  

    # Title
    parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')
    
    # File paths
    parser.add_argument('--inputPath', type=str, default="./Data/AEC/Videos/SEGMENTED_SIGN/",
                        help='relative path of images input.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')

    parser.add_argument('--dict_output', type=str, default="./Data/AEC/Dataset/dict/",
                        help='relative path of scv output set of landmarks.' +' Default: ./Data/AEC/Dataset/dict/')

    parser.add_argument('--keypoints_output', type=str, default="./Data/AEC/Dataset/keypoints/",
                        help='relative path of csv output set of landmarks.' + ' Default: ./Data/AEC/Dataset/keypoints/')

    # verbose
    parser.add_argument("--verbose", type=int, help="Verbosity")

    args = parser.parse_args()
    
    ####
    # Check if route is a folder. If so, create results for each of the videos
    # If no, create results for only the video passed
    #
    ###

    # Folder list of videos's frames
    if os.path.isdir(args.inputPath):
        folder_list = [file for file in os.listdir(args.inputPath)
                       if os.path.isdir(args.inputPath+file)]
        print("InputPath is Directory\n")
    else:
        folder_list = [args.inputPath]

    print("Folder List:\n")
    print(folder_list)

    #uv.createFolder(args.img_output)
    uv.createFolder(args.dict_output)
    uv.createFolder(args.keypoints_output)

    IdCount = 1
    LSP = []
    video_errors = []

    dictPath = args.dict_output+'/'+"dict"+'.json'

    dataList = []

    # Iterate over the folders of each video in Video/Segmented_gesture
    for videoFolderName in folder_list:
        print()
        videoFolderPath = args.inputPath + videoFolderName

        videoFolderList = [file for file in os.listdir(videoFolderPath)]

        cropVideoPath = '/'.join(args.inputPath.split('/')[0:-2])+'/cropped/'+videoFolderName+'/'
        uv.createFolder(cropVideoPath,createFullPath=True)

        for videoFile in videoFolderList:
            word = videoFile.split("_")[0]
            #if word not in ["G-R", "bien", "comer", "cuánto", "dos", "porcentaje", "proteína", "sí", "tú", "yo"]:
            #    continue

            keypointsDict = []

            videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

            pklKeypointsPath = args.keypoints_output+str(IdCount)+'.pkl'
            
            dataList.append([videoFolderPath,
                             word,
                             IdCount,
                             cropVideoPath,
                             pklKeypointsPath,
                             videoFolderName,
                             videoFile,
                             videoFolderPath])
            
            IdCount = IdCount + 1
    
    poolHandler(cores=-1, data=dataList)
