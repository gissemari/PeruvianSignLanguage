"""For every video file, saves an additional file signerX_sampleY_nframes
which contains a single integer (in text) with the number of frames."""
import glob
import os
import pandas as pd
import numpy as np

def main():
    input_dir = 'project/data/mp4'

    TRAIN_FILE = 'data/train_nframes.csv'
    VAL_FILE = 'data/val_nframes.csv'
    TEST_FILE = 'data/test_nframes.csv'

    for dataset in ['train', 'val']:
        videos = glob.glob(os.path.join(input_dir, dataset, '*_color.mp4'))

        if dataset == 'train':
            file = TRAIN_FILE
        if dataset == 'val':
            file = VAL_FILE

        nFrames = []
        videoName = []

        with open(file, encoding='utf-8') as orig_file:

            reader = pd.read_csv(orig_file, encoding='utf-8', header=None)
            nFrames = list(reader[1])
            videoName = list(reader[0])

        for video_file in videos:

            word =  "_".join(video_file.split('/')[-1].split('_')[0:2])

            for pos, name in enumerate(videoName):
                if name == word:
                    count = nFrames[pos] 
            '''
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
            '''
            #frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{count}\n')


if __name__ == '__main__':
    main()
