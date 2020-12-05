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

path = "./imgOut/mediapipe"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

path = "./jsonOut"
if not os.path.isdir(path):
    print("Directory %s has successfully created" % path)
    os.mkdir(path)

path = "./jsonOut/mediapipe"
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
# then double click in "getBaseModels" in openpose/models and wait
# Then uncomment and use the following code
# (REMEMBER: change the absolute path of OpenPoseDemo.exe in openposeLooper.py)


script_descriptor = open("./openPoseLooper.py")
openPose_script = script_descriptor.read()
print("\nRunning openPose models...")
exec(openPose_script)
print("openPose models - finished!")
