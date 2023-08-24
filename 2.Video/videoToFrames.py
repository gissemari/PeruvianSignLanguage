# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 16:23:57 2020

@author: Joe
"""
import argparse
import os
import cv2
import time

#########################
# ARGS
##############

parser = argparse.ArgumentParser(description='Video To frames ' +
                                 '(FaceMesh, Hands, Pose)')

# Path to folder with videos
parser.add_argument('--vid_input', type=str,
                    default="./Data/Videos/OnlySquare/",
                    help='relative path of images input.' +
                    ' Default: ./Data/Videos/OnlySquare/')

# Location of extracted images
parser.add_argument('--img_output', type=str,
                    default="./Data/Videos/OnlySquare/frames/",
                    help='relative path of images output with landmarks.' +
                    ' Default: ./Data/Videos/OnlySquare/frames/')

args = parser.parse_args()

#########################
# CONFIGURATION
##############

# Extract one frame each x frames
# Change this value - TODO make it configurable from args
one_frame_each = 1000

# Video file names
file_list = os.listdir(args.vid_input)

# extensions not allowed in File list loop
valid_extensions = [".mp4"]

#########################
# FILE LIST LOOP
##############
for idx, file in enumerate(file_list):

    # variables
    start_time = time.time()
    count = 0
    selectedFrames = 0
    success = True

    file_ext = os.path.splitext(file)[1]
    file_name = os.path.splitext(file)[0]

    # If the file does not have a valid file extension then skip it
    if (file_ext not in valid_extensions):
        continue

    # Read video
    vidcap = cv2.VideoCapture(args.vid_input + file)

    folderName = file_name + '/'

    if not os.path.isdir(args.img_output + folderName):
        print("Directory %s has successfully created" %
              (args.img_output + folderName))
        os.mkdir(args.img_output + folderName)

    while(success):

        # checks frame number and keeps one_frame_each
        if (count % one_frame_each == 0):
            selectedFrames += 1
            # reads next frame
            success, image = vidcap.read()

            # saves images to frame folder
            cv2.imwrite("%s%s%s_%d.png" %
                        (args.img_output, folderName, file_name, count),
                        image)

        else:
            # reads next frame
            success, image = vidcap.read()

        # (while) loops counter
        count += 1

    # Print file results
    print(selectedFrames, 'frames <--', file,
          "(%s seconds)" % str(time.time()-start_time))
