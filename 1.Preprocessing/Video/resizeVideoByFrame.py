# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 01:42:39 2021

@author: Joe
"""

# Standard library imports
import argparse
import os
import sys

# Third party imports
import cv2
import mediapipe as mp
import numpy as np

# Local imports
sys.path.append(os.getcwd())
import utils.video as uv

#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')

# File paths
parser.add_argument('--inputPath', type=str, default="./Data/AEC/Videos/SEGMENTED_SIGN/",
                    help='relative path of images input.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')

parser.add_argument('--outputPath', type=str, default="./Data/AEC/Videos/cropped/", help='relative path of video output.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')

# verbose
parser.add_argument("--verbose", type=int, help="Verbosity")

args = parser.parse_args()

#########################
# MODELS(Mediapipe)
#
# -Holistic
##############

print("\n#####\nHolistic Model\n#####\n")
mp_holistic = mp.solutions.holistic

#########################
# MODELS PARAMETERS
##############

# HOLISTIC parameters.
holistic = mp_holistic.Holistic(static_image_mode=False,
                                model_complexity=2,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

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

IdCount = 1
LSP = []
video_errors = []


# Iterate over the folders of each video in Video/Segmented_gesture
for videoFolderName in folder_list:
    print()
    videoFolderPath = args.inputPath + videoFolderName

    videoFolderList = [file for file in os.listdir(videoFolderPath)]

    cropVideoPath = args.outputPath + videoFolderName+'/'
    uv.createFolder(cropVideoPath,createFullPath=True)

    for videoFile in videoFolderList:

        word = videoFile.split("_")[0]
        #if word not in ["G-R", "bien", "comer", "cuánto", "dos", "porcentaje", "proteína", "sí", "tú", "yo"]:
        #    continue

        keypointsDict = []

        videoSegFolderName = videoFolderPath+'/'+videoFile[:-4]

        # Create a VideoCapture object
        cap = cv2.VideoCapture(videoFolderPath+'/'+videoFile)

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Check if camera opened successfully
        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoFolderPath+'/'+videoFile)
            video_errors.append(videoFolderPath+'/'+videoFile)
            continue

        #video = cv2.VideoWriter(cropVideoPath + word + '_' + str(IdCount)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))
        video = cv2.VideoWriter(cropVideoPath + word + '_' + str(IdCount)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))
        print(cropVideoPath + word + '_' + str(IdCount)+'.mp4',"Processing...")
        idx = 0

        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

                if(botton > height):
                    botton = height-1
                if(top < 0):
                    top = 0
                if(left < 0):
                    left = 0
                if(right > width):
                    right = width-1

                black = [0,0,0]
                imageBGR = cv2.copyMakeBorder(imageBGR[top:botton,left:right],0,0,0,0,cv2.BORDER_CONSTANT,value=black)
            else:
                if top == 0 and botton ==0 and left == 0 and right == 0:
                    print("NO KEYPOINT IN THE FIRST FRAME:",videoFolderPath + os.sep + videoFile)
                else:
                    # if it is not the first frame it takes the previous value of top, botton, left and right
                    imageBGR = cv2.copyMakeBorder(imageBGR[top:botton,left:right],0,0,0,0,cv2.BORDER_CONSTANT,value=black)

            imageBGR = cv2.resize(imageBGR, (220, 220))
            imageBGR = cv2.cvtColor(imageBGR, cv2.COLOR_RGB2BGR)

            video.write(imageBGR)
            # Next frame
            ret, frame = cap.read()
        video.release()

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
holistic.close()
