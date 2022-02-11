# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:55:55 2022

@author: Joe
"""

# Standard library imports
import argparse
import os
import random
from collections import Counter

# Third party imports
import pandas as pd
import pickle as pkl
import numpy as np
import csv

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
# Local imports

SEED = 52

parser = argparse.ArgumentParser(description='X and Y of keypoints and image Dataset distribution')

# Path to folder of the dictionary
parser.add_argument('--dict_Path_1', type=str,
                    default="./Data/AEC/Dataset/dict/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/dict/dict.json')

parser.add_argument('--dict_Path_2', type=str,
                    default="./Data/PUCP_PSL_DGI156/Dataset/dict/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/Dataset/dict/dict.json')


# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./Data/Dataset/readyToRun/",
                    help='relative path of dataset output.' +
                    ' Default: ./Data/Dataset/readyToRun/')

parser.add_argument('--words_File', type=str, default='', #./Data/list5.csv
                    help='relative path of words file' + ' Default: ./Data/')

# Number of top words
parser.add_argument("--words", type=int, default=10,
                    help="Number of top words")

args = parser.parse_args()


if args.words_File:
    wordList = pd.read_csv(args.words_File, header=None)
    print("WordList: ", wordList,"\n")

#Considering only the top k words in parameter --words
numWords = args.words

################################
#FIRST DICT
# We get the list of words-gloss from the json file
glossList_1 = pd.read_json(args.dict_Path_1)

# Saving post or index of the word (that will become the label) of all the instances found in dict.json
wordList_1 = dict(zip(wordList.iloc[:numWords,0], wordList.iloc[:numWords,1]))

timeStepDict_1 = {}

for glossIndex in glossList_1:

    word = str.upper(glossList_1[glossIndex]["gloss"])

    if args.words_File != '' and word not in wordList_1:
        continue
        
    for pos, instance in enumerate(glossList_1[glossIndex]["instances"]):
    
        # Saving time steps, more used in ResNET
        timestep = glossList_1[glossIndex]["instances"][pos]["frame_end"]
        if timestep > 25.0:
            print("INSTANCES ",instance["unique_name"])
        if word in timeStepDict_1:
            timeStepDict_1[word] = timeStepDict_1[word] + [timestep]
        else:
            timeStepDict_1.update({word: [timestep]})

#print("%%%%%%",timeStepDict_1)

#SECOND DICT
# We get the list of words-gloss from the json file
glossList_2 = pd.read_json(args.dict_Path_2)

# Saving post or index of the word (that will become the label) of all the instances found in dict.json
wordList_2 = dict(zip(wordList.iloc[:numWords,0], wordList.iloc[:numWords,1]))

timeStepDict_2 = {}

for glossIndex in glossList_2:

    word = str.upper(glossList_2[glossIndex]["gloss"])

    if args.words_File != '' and word not in wordList_2:
    #if args.words_File != '' and word not in ["HACER"]:
        continue
        
    for pos, instance in enumerate(glossList_2[glossIndex]["instances"]):
    
        # Saving time steps, more used in ResNET
        timestep = glossList_2[glossIndex]["instances"][pos]["frame_end"]
        #if timestep < 7.0:
            #print("INSTANCES ",instance["unique_name"])
        if word in timeStepDict_2:
            timeStepDict_2[word] = timeStepDict_2[word] + [timestep]
        else:
            timeStepDict_2.update({word: [timestep]})

# Create lists of Number of frames por sign

frameSizeList_1 = [val for group in timeStepDict_1.values() for val in group]
frameSizeList_2 = [val for group in timeStepDict_2.values() for val in group]

print("PUCP-stats")
print(np.quantile(frameSizeList_2, 0.95))

print(len([val for val in frameSizeList_2 if val > 46]))

print("AEC-stats")
print(np.quantile(frameSizeList_1, 0.95))

print(len([val for val in frameSizeList_1 if val > 29]))

print("\n\n\n")
print("AEC")
#print(frameSizeList_1)
print("######")
print("PUCP-DGI")
#print(frameSizeList_2)

# Plot
fig1, ax1 = plt.subplots()
bins = np.linspace(0,140,30)
ax1.hist(frameSizeList_1, bins, alpha=0.5, label='AEC', )
ax1.hist(frameSizeList_2, bins, alpha=0.5, label='PUCP-DGI', )
ax1.legend(loc='upper right')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Number of frame per sign')
ax1.set_title('Distribution of length in both datasets: PUCP-DGI and AEC')



frameCountList_1 = [len(group) for group in timeStepDict_1.values()]
frameCountList_2 = [len(group) for group in timeStepDict_2.values()]


#print(frameCountList_1)
#print(frameCountList_2)
fig2, ax2 = plt.subplots()
bins = np.linspace(0, 70, 10)
ax2.hist(frameCountList_1, bins, alpha=0.5, label='AEC')
#ax2.hist(frameCountList_2, bins, alpha=0.5, label='AEC')
ax2.legend(loc='upper right')
ax2.set_ylabel('Frequency')
ax2.set_xlabel('Number of frame per sign')
ax2.set_title('Distribution of length in both datasets: PUCP-DGI and AEC')


# Plot inside class
names_aec = [name for name, val in timeStepDict_1.items()]
values_aec = [val for name, val in timeStepDict_1.items()]

fig3, ax3 = plt.subplots()
bins = np.linspace(0, 45, 15)
ax3.hist(values_aec, bins, alpha=0.5, label=names_aec)
#ax2.hist(frameCountList_2, bins, alpha=0.5, label='AEC')
ax3.legend(loc='upper right')
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Number of frame per sign')
ax3.set_title('Distribution of length in both datasets: AEC')


names_pucp = [name for name, val in timeStepDict_2.items()]
values_pucp = [val for name, val in timeStepDict_2.items()]

fig4, ax4 = plt.subplots()
bins = np.linspace(0, 45, 15)
ax4.hist(values_pucp, bins, alpha=0.5, label=names_pucp, )
#ax2.hist(frameCountList_2, bins, alpha=0.5, label='AEC')
ax4.legend(loc='upper right')
ax4.set_ylabel('Frequency')
ax4.set_xlabel('Number of frame per sign')
ax4.set_title('Distribution of length in both datasets: PUCP-CGI')

