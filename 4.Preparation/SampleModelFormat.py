# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 22:35:36 2021

@author: Joe
"""

# Standard library imports
import argparse

# Third party imports
import numpy as np
import pickle as pkl

# Local imports
import utils.video as uv  # for folder creation

parser = argparse.ArgumentParser(description='X and Y Dataset reshaper')

# Number of Time steps
parser.add_argument("--timesteps", type=int, default=17,
                    help="Max number of timestep allowed")

# Path to the output folder
parser.add_argument('--output_Path', type=str,
                    default="./Data/Dataset/",
                    help='relative path of dataset output.' +
                    ' Default: ./Data/Dataset/')

args = parser.parse_args()


# Third party imports
import pickle

fileName = "toReshape/"

with open('./Data/Dataset/'+fileName+'X.data', 'rb') as f:
    X = pickle.load(f)

with open('./Data/Dataset/'+fileName+'X_timeSteps.data', 'rb') as f:
    X_timeSteps = pickle.load(f)

with open('./Data/Dataset/'+fileName+'Y.data', 'rb') as f:
    Y = pickle.load(f)

with open('./Data/Dataset/'+fileName+'weight.data', 'rb') as f:
    weight = pickle.load(f)

with open('./Data/Dataset/'+fileName+'Y_meaning.data', 'rb') as f:
    y_meaning = pickle.load(f)

for ind in range(len(Y)):

    if len(X[ind]) == args.timesteps:
        continue
    # To complete the number of timesteps if it is less than requiered
    elif len(X[ind]) < args.timesteps:

        for _ in range(args.timesteps - len(X[ind])):
            X[ind] = np.append(X[ind], [X[ind][-1]], axis=0)

    # More than the number of timesteps
    else:

        toSkip = len(X[ind]) - args.timesteps
        interval = len(X[ind]) // toSkip

        # Generate an interval of index
        a = [val for val in range(0, len(X[ind])) if val % interval == 0]

        # from the list of index, we erase only the number of index we want to skip
        X[ind] = np.delete(X[ind], a[-toSkip:], axis=0)

filePath = "readyToRun/"
uv.createFolder(args.output_Path+filePath)

with open(args.output_Path+filePath+'X.data', 'wb') as f:
    pkl.dump(X, f)

with open(args.output_Path+filePath+'Y.data', 'wb') as f:
    pkl.dump(Y, f)

with open(args.output_Path+filePath+'X_timeSteps.data', 'wb') as f:
    pkl.dump(X_timeSteps, f)

with open(args.output_Path+filePath+'weight.data', 'wb') as f:
    pkl.dump(weight, f)

with open(args.output_Path+filePath+'Y_meaning.data', 'wb') as f:
    pkl.dump(y_meaning, f)
        