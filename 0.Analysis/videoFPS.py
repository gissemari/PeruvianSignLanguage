# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 12:29:30 2022

@author: Joe
"""

# Standard library imports
import argparse
import os

# Third party imports
import cv2
import numpy as np
import pandas as pd

# Local imports
import utils.video as uv  # for folder creation and keypoints normalization


#########################
# ARGS
##############

# Title
parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')

# File paths
parser.add_argument('--inputPath_1', type=str, default="./Data/AEC/Videos/RawVideo/",
                    help='relative path of images input.' + ' Default: ./Data/AEC/Videos/RawVideo/')

parser.add_argument('--inputPath_2', type=str, default="./Data/PUCP_PSL_DGI156/Videos/original/",
                    help='relative path of images input.' + ' Default: ./Data/PUCP_PSL_DGI156/Videos/original/')

args = parser.parse_args()


####
# Check if route is a folder. If so, create results for each of the videos
# If no, create results for only the video passed
#
###

# Folder list of videos's frames
if os.path.isdir(args.inputPath_1):
    folder_list = [file for file in os.listdir(args.inputPath_1)]
else:
    folder_list = [args.inputPath_1]

video_errors = []

print("AEC")
# AEC
for videoFolderName in folder_list:

    videoFolderPath = args.inputPath_1 + videoFolderName

    # Create a VideoCapture object
    cap = cv2.VideoCapture(videoFolderPath)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Unable to read camera feed",videoFolderPath)
        video_errors.append(videoFolderPath)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("%8f %s "%(fps, videoFolderName))

print("##")

# Folder list of videos's frames
if os.path.isdir(args.inputPath_2):
    folder_list = [file for file in os.listdir(args.inputPath_2)]
else:
    folder_list = [args.inputPath_2]

print("PUCP-CGI")
# PUCP-DGI
for videoFolderName in folder_list:

    videoFolderPath = args.inputPath_2 + videoFolderName

    # Create a VideoCapture object
    cap = cv2.VideoCapture(videoFolderPath)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Unable to read camera feed",videoFolderPath)
        video_errors.append(videoFolderPath)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("%8f %s "%(fps, videoFolderName))

if video_errors:
    print("Errors found in: ",video_errors)