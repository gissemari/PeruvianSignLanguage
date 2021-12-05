"""For every video file, saves an additional file signerX_sampleY_nframes
which contains a single integer (in text) with the number of frames."""
import argparse
import glob
import os

import torchvision
import cv2
import numpy as np

def main():
    input_dir = 'project/data/mp4'
    for dataset in ['train', 'val', 'test']:
        videos = glob.glob(os.path.join(input_dir, dataset, '*_color.mp4'))
        for video_file in videos:

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

            #frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{count}\n')


if __name__ == '__main__':
    main()
