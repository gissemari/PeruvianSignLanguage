# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:51:46 2020

@author: Joe
"""
import sys
import os

#######################
# FROM VIDEO TO FRAMES
script_descriptor = open("./1.Preprocessing/Video/videoToFrames.py")
videoToFrames_script = script_descriptor.read()

print("Running videoToFrame script...")
exec(videoToFrames_script)
print("videoToFrame script - Finished!\n")

#######################
# CREATE MEDIAPIPE FOLDERS
path = "./imgOut"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

path = "./pklOut"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

#######################
# MEDIAPIPE
script_descriptor = open("./mediapipe/mediaPipe.py")
mediaPipe_script = script_descriptor.read()
sys.argv = ["mediaPipe.py", "--face", "--hands", "--pose"]

print("\nRunning mediaPipe models...")
exec(mediaPipe_script)
print("mediaPipe models - finished!")

#######################
# CREATE POSENET FOLDERS
path = "./jsonOut"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

#######################
# POSENET
# Download binaries from
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
# and unzip it in the root folder
# (Recommentadion: if has CUDA then use GPU version, if not use CPU version)
# Then uncomment this

'''
# Change relative path
os.chdir("./openpose")

# use this if you want to suppress output to stdout from the subprocess
path = 'D:/Documentos/Projects/PeruvianSignLanguaje/openpose/bin/'
program = 'OpenPoseDemo.exe '
direction = path + program

imageDir = "--image_dir ../Data/Videos/OnlySquare/frames/germinados/ "
models = "--face --hand "
jsonOut = "--write_json ../jsonOut "
options = "--display 0 "
# netRes = "--net_resolution 320x320" or (320x176) but both have lower accuracy
imageOut = "--write_images ../imgOut"
arguments = direction + imageDir + models + jsonOut + options + imageOut

os.system(direction + arguments)
'''
