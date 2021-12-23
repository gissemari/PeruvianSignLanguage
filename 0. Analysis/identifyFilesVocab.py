import sys
import argparse
import pysrt
import pandas as pd
import os
from collections import Counter

parser = argparse.ArgumentParser(description='Translating SRTs')
parser.add_argument('--srtPath', type=str, default='./../Data/PUCP_PSL_DGI156/SRT/SRT_SEGMENTED_SIGN/', help='Path where the original SRT is located')
parser.add_argument('--vocabFile', type=str, default='./../Data/list.csv', help='Input File Name')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
args = parser.parse_args()

#wordList looking for
numWords = 10
wordList = pd.read_csv(args.vocabFile, header=None)
wordList = dict(zip(wordList.iloc[:numWords,0], wordList.iloc[:numWords,1]))
print("WordList: ", wordList)

#list of files that contain words in the vocabulary
listFilesWithVocab = {}

#Listing SRT files in folder
inputName = args.inputName
srtPath = args.srtPath
if inputName == '':
    listFile = [file for file in os.listdir(srtPath) if os.path.isfile(srtPath+file)]
else:
    listFile = [inputName]

for srtFileName in listFile:
    
    # Read SRT
    srtFile = pysrt.open(srtPath+srtFileName, encoding='utf-8')
    newDict = {}
    flgAtLeastOne = False
    for line in srtFile:
        wordSRT =  line.text.upper()
        #print(wordSRT)
        if wordSRT in wordList:
            flgAtLeastOne = True
            if wordSRT in newDict:
                newDict[wordSRT] +=1
            else:
                newDict[wordSRT] = 1
    print(srtFileName, flgAtLeastOne)
    vocabFreqInFile = Counter(newDict)
    if flgAtLeastOne: #or newDict not empty
        listFilesWithVocab[srtFileName] = vocabFreqInFile

    df = pd.DataFrame.from_dict(listFilesWithVocab, orient='index')
print(df)
df.to_csv('vocabPerFile.csv')