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
            #''' next comment - gissella
            count = 0
            word =  "_".join(nameVideo.split('_')[0:2])
            #word =  video_file.split('/')[-1][:-4]

            print(word)
            for pos, name in enumerate(videoName):
                if name == word:
                    count = nFrames[pos]
                    print(count)
            #end next comment - gissella '''
                    
            ''' prev comment
            cap = cv2.VideoCapture(video_file)
            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Unable to read camera feed", video_file)
                continue
            ret, frame = cap.read()
            frames = []
            count = 0
            while ret is True:
                frm = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frm)
                count += 1
                ret, frame = cap.read()
            
            end prev comment '''
            
            #frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{count}\n')


if __name__ == '__main__':
    main()
