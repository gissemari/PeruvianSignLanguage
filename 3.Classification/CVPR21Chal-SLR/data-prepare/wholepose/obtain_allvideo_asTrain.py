
import os
import shutil

base_dir = '../../../../Data/AEC/Videos/SEGMENTED_SIGN/'
target_dir = '../data/sign/train/'

rawVideoNames = os.listdir(base_dir)

for rawVideoN in rawVideoNames:
    video_dir = base_dir + rawVideoN + '/'
    video_list = os.listdir(video_dir)

    for videoN in video_list:
        videoName = videoN.split('.')[0]
        videoName = videoName.upper()

        videoOriginPath = video_dir + videoN
        videoTargetPath = target_dir + videoName + '_color.mp4'

        shutil.copyfile(videoOriginPath, videoTargetPath)
