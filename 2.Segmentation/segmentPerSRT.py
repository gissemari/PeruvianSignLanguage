import cv2
import numpy as np
import argparse
import pysrt
import nltk
import utils.video as uv

from os import listdir
from os.path import isfile, join, exists
from scenedetect.frame_timecode import FrameTimecode

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--rawVideoPath', type=str, default='./../Data/RawVideo/', help='Path where per-line files are located')
parser.add_argument('--srtPath', type=str, default='./../Data/SRT/SRT_gestures/', help='Path where per-line files are located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
parser.add_argument('--outputVideoPath', type=str, default='./../Data/OnlySquare/', help='Path where per-line files are located')
parser.add_argument('--fpsOutput', type=int, default=25, metavar='fpsO',help='Frames per second for the output file')


args = parser.parse_args()

srtPath = args.srtPath
rawVideoPath = args.rawVideoPath
inputName = args.inputName
outputVideoPath = args.outputVideoPath
fpsOutput = args.fpsOutput

# Create a VideoCapture object
cap = cv2.VideoCapture(rawVideoPath+inputName+'.mp4')

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")


fps = uv.getNumFrames(cap)


# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


### When defining VideoWriter (width, height)
### When cropping the frames (height, width)
videoWidth = 220
videoHeight = 220


### X1,Y1 .... X2, Y1
### X1,Y2 .... X2, Y2
x1 = 380
x2 = x1 + videoHeight + 1#601
y1 = 988
y2 = y1 + videoWidth + 1

count = 0
#Set begining of reading
#cap.set(cv2.CAP_PROP_POS_MSEC,1000)



# Read SRT
srtOriginal = pysrt.open(srtPath+inputName+'.srt', encoding='utf-8')#, encoding='iso-8859-1'

### Iterate over the SRT

sentence = 0
for line in srtOriginal[:10]:
    #SubRipItem .strftime((“%H:%M:%S.ff”) #“%H:%M:%S[.nnn]”)
    #print(line.start.to_time(),line.end.to_time())
    ini =  FrameTimecode(timecode = line.start.to_time().strftime("%H:%M:%S.%f"), fps = fps)
    iniFrame = ini.get_frames()
    end =  FrameTimecode(timecode = line.end.to_time().strftime("%H:%M:%S.%f"), fps = fps)
    endFrame = end.get_frames()

    '''
    positionStart = line.start.hours*60*60 + line.start.minutes*60 + line.start.seconds + line.start.frame*0.000001
    positionStart = positionStart*fps
    positionEnd = line.end.hours*60*60 + line.end.minutes*60 + line.end.seconds  + line.end.frame*0.000001
    positionEnd = positionEnd*fps
    print(iniFrame, endFrame, positionStart,positionEnd)
    '''
    print(line.start.to_time().strftime("%H:%M:%S.%f"), iniFrame, endFrame)
    outSegment = cv2.VideoWriter(outputVideoPath+inputName+'/'+str(sentence)+'_'+line.text+'.avi', fcc, fpsOutput, (videoWidth, videoHeight))

    # Doc: CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
    # cap.set(cv2.CAP_PROP_POS_MSEC,line.start.to_time())
    # To give a threshold
    for i in range(iniFrame-1, endFrame+1):
        
        pos = i/fps*1000 # Frame number divided by rate to obtain the second and multiply by miliSecs
        cap.set(cv2.CAP_PROP_POS_MSEC,pos)

        ret, frame = cap.read()
        # While a frame was read
        if ret == True:
            crop_frame = frame[x1:x2, y1:y2]
            outSegment.write(crop_frame)

    sentence +=1
    outSegment.release()

# When everything done, release the video capture and video write objects
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
