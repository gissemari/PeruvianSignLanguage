# Standard library imports
import argparse
import os

# Third party imports
import cv2
import pysrt
from scenedetect.frame_timecode import FrameTimecode

# Local imports
import utils.video as uv

# import nltk
# import numpy as np
# from os import listdir, mkdir
# from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--rawVideoPath', type=str,
                    default='./Data/Videos/RawVideo/',
                    help='Path where per-line files are located')
parser.add_argument('--srtPath', type=str,
                    default='./Data/SRT/SRT_gestures/',
                    help='Path where per-line files are located')
parser.add_argument('--inputName', type=str, default='',
                    help='Input File Name')
parser.add_argument('--outputVideoPath', type=str,
                    default='./Data/Videos/Segmented_gestures/',
                    help='Path where per-line files are located')

# parser.add_argument('--fpsOutput', type=int, default=25, metavar='fpsO', help='Frames per second for the output file')
parser.add_argument('--flgGesture', type=int, default=1, metavar='FLGES', help='Frames per second for the output file')
parser.add_argument('--width', type=int, default=220, metavar='WIDTH', help='Width of SL signer or interpreter')
parser.add_argument('--height', type=int, default=220, metavar='HEIGHT', help='Height of SL signer or interpreter')
parser.add_argument('--x1', type=int, default=380, metavar='X1', help='Beginning of coordinate x frame')
parser.add_argument('--y1', type=int, default=988, metavar='Y1', help='Beginning of coordinate y frame')


args = parser.parse_args()

srtPath = args.srtPath
rawVideoPath = args.rawVideoPath
inputName = args.inputName
outputVideoPath = args.outputVideoPath
# fpsOutput = args.fpsOutput
flgGesture = args.flgGesture

# ## When defining VideoWriter (width, height)
# ## When cropping the frames (height, width)
videoWidth = args.width
videoHeight = args.height


# ## X1,Y1 .... X2, Y1
# ## X1,Y2 .... X2, Y2
# AEC dataset x1 380 y1 988
# PUCP DGI dataset ...
x1 = args.x1
x2 = x1 + videoHeight + 1  # 601
y1 = args.y1
y2 = y1 + videoWidth + 1

count = 0
# Set begining of reading
# cap.set(cv2.CAP_PROP_POS_MSEC,1000)


if inputName == '':
    listFile = [file for file in os.listdir(srtPath)
                if os.path.isfile(srtPath+file)]
else:
    listFile = [inputName]

print(srtPath, inputName, listFile)

for filePath in listFile:

    # to get only the name without the path
    inputName = os.path.basename(filePath)
    inputName = os.path.splitext(inputName)[0]
    outputFolder = outputVideoPath+inputName
    outputFolder = outputFolder.replace(' ','_').replace('(','').replace(')','')
    if uv.createFolder(outputFolder, createFullPath=True):
        print('Created folder :', outputFolder)
    else:
        print('Folder existed :', outputFolder)

    # Create a VideoCapture object
    cap = cv2.VideoCapture(rawVideoPath+inputName+'.mp4')

    # Check if camera opened successfully
    if(cap.isOpened() is False):
        print("Unable to read camera feed", rawVideoPath+inputName+'.mp4')

    fps = uv.getNumFrames(cap)
    fpsOutput = fps
    print(fps)
    # Define the codec and create VideoWriter object.The output
    # is stored in 'outpy.avi' file.
    fcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'M', 'J', 'P', 'G' # *'MP4V'

    # Read SRT
    #                                                  encoding='iso-8859-1'
    srtOriginal = pysrt.open(srtPath+inputName+'.srt', encoding='utf-8')

    # ## Iterate over the SRT

    sentence = 0
    for line in srtOriginal:
        # SubRipItem .strftime(("%H:%M:%S.ff"") #"%H:%M:%S[.nnn]"s)
        # print(line.start.to_time(),line.end.to_time())
        ini = FrameTimecode(timecode=line.start.to_time().strftime(
            "%H:%M:%S.%f"), fps=fps)
        iniFrame = ini.get_frames()
        end = FrameTimecode(timecode=line.end.to_time().strftime(
            "%H:%M:%S.%f"), fps=fps)
        endFrame = end.get_frames()

        '''
        positionStart = line.start.hours*60*60 + line.start.minutes*60 +
                        line.start.seconds + line.start.frame*0.000001
        positionStart = positionStart*fps
        positionEnd = line.end.hours*60*60 + line.end.minutes*60 +
                      line.end.seconds  + line.end.frame*0.000001
        positionEnd = positionEnd*fps
        print(iniFrame, endFrame, positionStart,positionEnd)
        '''
        print(line.start.to_time().strftime("%H:%M:%S.%f"), iniFrame, endFrame, endFrame - iniFrame)
        if flgGesture:
            #line.text.upper()
            outSegment = cv2.VideoWriter(outputFolder+'/'+ line.text+'_'+str(sentence)+'.mp4', fcc, fpsOutput, (videoWidth, videoHeight))
        else:
        	
            #foldName = outputVideoPath+inputName+'/'+inputName+'_'+str(sentence)+'.mp4'
            #print(foldName)
            outSegment = cv2.VideoWriter(outputVideoPath+inputName+'/'+str(sentence+1)+'.mp4', fcc, fpsOutput, (videoWidth, videoHeight))
        # Doc: CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
        # cap.set(cv2.CAP_PROP_POS_MSEC,line.start.to_time())
        # To give a threshold
        for i in range(iniFrame-1, endFrame+1):

            # Frame number divided by rate to obtain the second and
            # multiply by miliSecs
            pos = i/fps*1000
            cap.set(cv2.CAP_PROP_POS_MSEC, pos)

            ret, frame = cap.read()
            # While a frame was read
            if ret is True:
                crop_frame = frame[x1:x2, y1:y2]
                outSegment.write(crop_frame)

        sentence += 1
        outSegment.release()

    # When everything done, release the video capture and video write objects
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
