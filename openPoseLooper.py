# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 18:27:46 2020

@author: Joe
"""
import os

os.chdir("./openpose")

absolutePath = 'D:/Documentos/Projects/PeruvianSignLanguaje/openpose/bin/'
program = 'OpenPoseDemo.exe '
direction = absolutePath + program

models = "--face --hand "
options = "--display 0 "
# netRes = "--net_resolution 320x320" or (320x176) but both have lower accuracy

folder_list = os.listdir("../Data/Videos/OnlySquare/frames/")

path = '../jsonOut/openPose'
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

path = '../imgOut/openPose'
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

for folder in folder_list:

    imageDir = "--image_dir ../Data/Videos/OnlySquare/frames/%s/ " % folder
    jsonOut = "--write_json ../jsonOut/openPose/%s " % folder
    imageOut = "--write_images ../imgOut/openPose/%s " % folder

    arguments = direction + imageDir + models + jsonOut + options + imageOut

    path = '../jsonOut/openPose/' + folder
    if not os.path.isdir(path):
        print("Directory %s has successfully created" % path)
        os.mkdir(path)

    path = '../imgOut/openPose/' + folder
    if not os.path.isdir(path):
        print("Directory %s has successfully created" % path)
        os.mkdir(path)

    os.system(direction + arguments)
