import numpy as np
import argparse
import pysrt
import nltk
import pickle as pkl
import os
from collections import Counter

#from os import listdir, mkdir
#from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--EnglishWord', type=str, default='', help='Word for which to analyze the distance within class')
parser.add_argument('--configFile', type=str, default='config.txt', help='List of Words for which to analyze the distance within class')
parser.add_argument('--pklPath', type=str, default="./../../Data_cLSP/Data/Keypoints/pkl/Segmented_gestures/")
parser.add_argument('--outputVideoPath', type=str, default='./../Data/Videos/Segmented_gestures/', help='Path where per-line files are located')
#parser.add_argument('--fpsOutput', type=int, default=25, metavar='fpsO',help='Frames per second for the output file')
parser.add_argument('--flgGesture', type=int, default=1, metavar='FLGES',help='Frames per second for the output file')


args = parser.parse_args()


if args.EnglishWord =='':
    with open(args.configFile, encoding='utf-8') as f:
        listWords = f.readlines()
else:
    listWords = [args.EnglishWord]

for word in listWords:
    
    listLen = []
    listFolders = os.listdir(args.pklPath)
    listPklFiles = []
    for folder in listFolders:
        if not os.path.isdir(args.pklPath+folder):
            continue
        listFiles = os.listdir(args.pklPath+folder)
        #print(listFiles)
        for file in listFiles:
            ## find the underscore
            lenWord = file.find('_')

            if file[:lenWord]==word[:-1]:
                #print(file[:lenWord], word)
                pklFile = open(args.pklPath+folder+'/'+file, 'rb')
                listPklFiles.append(file)
                x = pkl.load(pklFile)
                listLen.append(x.shape[0])
    # First analyze length of instances
    arrSequences = np.array(listLen)
    print(word,arrSequences.mean(), arrSequences.std() )
    #print(listPklFiles)