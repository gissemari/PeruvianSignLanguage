# Standard library imports
import argparse
import os
import sys

# Third party imports
import cv2
import pysrt
from scenedetect.frame_timecode import FrameTimecode

# Local imports
sys.path.append(os.getcwd())
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
parser.add_argument('--width', type=int, default=-1, metavar='WIDTH', help='Width of SL signer or interpreter')
parser.add_argument('--height', type=int, default=-1, metavar='HEIGHT', help='Height of SL signer or interpreter')
parser.add_argument('--x1', type=int, default=-1, metavar='X1', help='Beginning of coordinate x frame')
parser.add_argument('--y1', type=int, default=-1, metavar='Y1', help='Beginning of coordinate y frame')


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

crop = False
#if this four variables are not set as default 
if (args.x1 != -1 and args.y1 != -1 and args.width != -1 and args.height != -1):
    crop = True 

if crop:
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
    if not crop:
        videoWidth  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        videoHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

    fps = uv.getNumFrames(cap)
    fpsOutput = fps

    totalFrameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of video's frames
    videoDuration = totalFrameCount/fps # Video time in seconds
    print("N° frames:",totalFrameCount,"Duration:",videoDuration)

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
        
        #print(line.start.to_time().strftime("%H:%M:%S.%f"),line.end.to_time().strftime("%H:%M:%S.%f"), ini, end, iniFrame, endFrame, endFrame - iniFrame)
        
        if flgGesture:
            #line.text.upper()
            rmSpacesName = line.text.replace(' ','-')
            outSegment = cv2.VideoWriter(outputFolder+'/'+ rmSpacesName +'_'+str(sentence)+'.mp4', fcc, fpsOutput, (videoWidth, videoHeight))
        else:
        	
            #foldName = outputVideoPath+inputName+'/'+inputName+'_'+str(sentence)+'.mp4'
            #print(foldName)
            outSegment = cv2.VideoWriter(outputVideoPath+inputName+'/'+str(sentence+1)+'.mp4', fcc, fpsOutput, (videoWidth, videoHeight))
        # Doc: CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
        # cap.set(cv2.CAP_PROP_POS_MSEC,line.start.to_time())
        # To give a threshold
        for i in range(iniFrame, endFrame+1):

            # from https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
            # frame_no = (frame_seq /(time_length*fps))
            # where
            # frame_no: 0-based index of the frame to be decoded/captured (range 0.0-1.0)
            # i: frame we select in range ("-1" is added because frame count starts at 0 value)
            # fps: frames per seconds of the video
            # videoDuration: duration in seconds of the video
            
            # it seems like we are using calculating pos in vaine, we can use i
            #pos = i-1/fps*videoDuration
            #print(i, pos, i-1/fps*1000)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = cap.read()
            # While a frame was read
            if ret is True:
                if crop:
                    crop_frame = frame[x1:x2, y1:y2]
                else:
                    crop_frame = frame
                outSegment.write(crop_frame)

        sentence += 1
        outSegment.release()

    # When everything done, release the video capture and video write objects
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
