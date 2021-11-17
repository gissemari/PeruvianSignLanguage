"""For every video file, saves an additional file signerX_sampleY_nframes
which contains a single integer (in text) with the number of frames."""
import argparse
import glob
import os
import cv2
import torchvision


def main(args):

    videos = [file for file in os.listdir(args.input_dir)]

    for video_file in videos:
        print(video_file)
        cap = cv2.VideoCapture(args.input_dir+'/'+video_file)

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
        with open(video_file.replace('.mp4', 'nframes'), 'w') as of:
            of.write(f'{count}\n')

def fixName(args):
    videos = glob.glob(os.path.join(args.input_dir, '*.mp4'))
    for video_file in videos:
        frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
        print(frames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()
    main(args)
