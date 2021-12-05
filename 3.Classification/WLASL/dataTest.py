#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 17:28:25 2021

@author: joe
"""
import argparse
import os

from tkinter import *
import pandas as pd

parser = argparse.ArgumentParser(description='X and Y Dataset generator')

parser.add_argument('--main_folder_Path', type=str,
                    default="data/pose_per_individual_videos/")

args = parser.parse_args()


canvas_width = 256
canvas_height = 256


def create_circle(x, y, r, canvasName): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1)

master = Tk()
master.title("Points")
w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack(fill=BOTH)

foldersToLoad = os.listdir(args.main_folder_Path)

wordList = []
timeStepDict = {}

count = 1

exclude = list(range(23,33))

denied = set([9 , 10, 11, 12, 13, 14, 19, 20, 21, 22]) # 23 and 24 are zeros
toChange = set(range(1,11))


for folderName in foldersToLoad:
    
    folder = os.listdir(args.main_folder_Path+folderName)

    for file in folder:
        
        if(count):
            count = count -1
        else:
            continue
        
        fileData = pd.read_json(args.main_folder_Path+folderName+'/'+file)
        data = fileData['people'][0]

        x = [v for i, v in enumerate(data['pose_keypoints_2d']) if i % 3 == 0 and i // 3 not in exclude]
        y = [v for i, v in enumerate(data['pose_keypoints_2d']) if i % 3 == 1 and i // 3 not in exclude]

        x = x + [0.0, 0.0]
        y = y + [0.0, 0.0]

        inter = denied & toChange        
        
        inter = list(inter)
        denied = list(denied)
        toChange = list(toChange)

        denied = [val for val in denied if val not in inter]
        toChange = [val for val in toChange if val not in inter]

        for pos in range(len(toChange)):
            if pos in inter:
                continue
            tmp = x[toChange[pos]]
            x[toChange[pos]] = x[denied[pos]]
            x[denied[pos]] = tmp
            
            tmp = y[toChange[pos]]
            y[toChange[pos]] = y[denied[pos]]
            y[denied[pos]] = tmp
            
        for pos in range(len(x)):
            if pos in exclude:
                continue
            if pos in denied:
                create_circle(x[pos],y[pos],0.5,w)






