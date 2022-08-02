"""For every video file, saves an additional file signerX_sampleY_nframes
which contains a single integer (in text) with the number of frames."""
import argparse
import glob
import os
import csv
import torchvision
import cv2
import numpy as np
from sys import platform
import argparse

parser = argparse.ArgumentParser(description='Classification')
parser.add_argument('--train', type=int, default=1, help='Create files for train or not, for test')

args = parser.parse_args()

def main():
    input_dir = 'project/data/mp4'

    TRAIN_FILE = 'data/train_nframes.csv'
    VAL_FILE = 'data/val_nframes.csv'
    TEST_FILE = 'data/test_nframes.csv'
    
    if args.train==1:
        listSplits = ['train', 'val']
    else:
        listSplits = ['test']

    for dataset in listSplits:
        videos = glob.glob(os.path.join(input_dir, dataset, '*_color.mp4'))

        if dataset == 'train':
            file = TRAIN_FILE
        if dataset == 'val':
            file = VAL_FILE
        if dataset == 'test':
            file = TEST_FILE
        

        print(dataset)

        jobCount = 0

        nFrames = []
        videoName = []

        with open(file, encoding='utf-8') as orig_file:

            reader = csv.reader(orig_file)
            for row in reader:
                nFrames.append(row[1])
                videoName.append(row[0])


        for video_file in videos:
        
            if platform == 'linux' or platform == 'linux2':
                nameVideo = video_file.split('/')[-1]
            else:
                nameVideo = video_file.split('\\')[-1]
            # next comment - gissella


            word =  "_".join(nameVideo.split('_')[0:-1])
            #word =  video_file.split('/')[-1][:-4]

            for pos, name in enumerate(videoName):
 
                if name == word:
                    
                    count = nFrames[pos]
                    jobCount += 1
                    #print(word, count)
            #end next comment - gissella
            
            #frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{count}\n')
        print(jobCount)


if __name__ == '__main__':
    main()
