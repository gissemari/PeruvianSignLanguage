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
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{count}\n')


if __name__ == '__main__':
    main()
