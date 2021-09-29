import numpy as np
import argparse
import pysrt
import nltk
import os
from collections import Counter

#from os import listdir, mkdir
#from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--srtPath', type=str, default='./../Data/SRT/SRT_gestures/', help='Path where per-line files are located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
parser.add_argument('--outputVideoPath', type=str, default='./../Data/Videos/Segmented_gestures/', help='Path where per-line files are located')
#parser.add_argument('--fpsOutput', type=int, default=25, metavar='fpsO',help='Frames per second for the output file')
parser.add_argument('--flgGesture', type=int, default=1, metavar='FLGES',help='Frames per second for the output file')


args = parser.parse_args()

srtPath = args.srtPath
inputName = args.inputName
outputVideoPath = args.outputVideoPath
#fpsOutput = args.fpsOutput
flgGesture = args.flgGesture



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


if inputName == '':
    listFile = [ file for file in os.listdir(srtPath) if os.path.isfile(srtPath+file)]
else:
    listFile = [inputName]


print(srtPath,inputName,listFile)

dictSigns = {}
words = []
sentence = 0


for filePath in listFile:

    inputName = os.path.basename(filePath) # to get only the name without the path
    inputName = os.path.splitext(inputName)[0]
    outputFolder = outputVideoPath+inputName

    # Read SRT
    srtOriginal = pysrt.open(srtPath+inputName+'.srt', encoding='utf-8')#, encoding='iso-8859-1'

    ### Iterate over the SRT
    
    for line in srtOriginal:
        words.append(line.text)
        #print(dictSigns.keys())
        if line.text in dictSigns.keys():
            dictSigns[line.text]+=1
        else:
            dictSigns[line.text] = 1

        sentence+=1

cnt = Counter(words)

print(len(dictSigns), sentence)
print(cnt)
