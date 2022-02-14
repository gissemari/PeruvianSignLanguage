import cv2
import os
import sys
import numpy as np
import argparse
sys.path.append(os.getcwd())
import utils.video as uv
from os import path

from os import listdir
from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--rawVideoPath', type=str, default='./../../Data/Videos/RawVideo/', help='Path where per-line files are located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
parser.add_argument('--outputVideoPath', type=str, default='./../../Data/Videos/OnlySquare/', help='Path where per-line files are located')

args = parser.parse_args()

rawVideoPath = args.rawVideoPath
inputName = args.inputName
outputVideoPath = args.outputVideoPath


# Create a VideoCapture object

cap = cv2.VideoCapture(rawVideoPath+inputName+'.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")


fps = uv.getNumFrames(cap)


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# 25 is the frame per second calculated by video_fps.py


### When defining videoWriter (width, height)
### When cropping the frames (height, width)
videoWidth = 220
videoHeight = 220

fcc = cv2.VideoWriter_fourcc(*'MP4V')#'M', 'J', 'P', 'G'
out = cv2.VideoWriter(outputVideoPath+inputName+'.mp4', fcc, fps, (videoWidth, videoHeight))
# 29 -> 29:37 de 28:40
# 29.97 -> 21.06 de 28:40

### X1,Y1 .... X2, Y1
### X1,Y2 .... X2, Y2
x1 = 380
x2 = x1 + videoHeight + 1#601
y1 = 988
y2 = y1 + videoWidth + 1

count = 0
#Set begining of reading
#cap.set(cv2.CAP_PROP_POS_MSEC,1000)

while(True):

    ret, frame = cap.read()

    # While a frame was read
    if ret == True:
        # Crop area is [x1:x1+videoHeight, y1:y1+videoWidth]
        crop_frame = frame[x1:x2, y1:y2]

        #out.write(frame)
        out.write(crop_frame)
        # Display the resulting frame
        # cv2.imshow('frame', crop_frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

    count +=1



# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
