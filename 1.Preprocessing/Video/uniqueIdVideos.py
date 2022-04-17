import argparse
import os

import pandas as pd
from shutil import copyfile

parser = argparse.ArgumentParser(description='X and Y of keypoints and image Dataset distribution')

parser.add_argument('--dict_Path', type=str,
                    default="./Data/AEC/dict.json",
                    help='relative path of keypoints input.' +
                    ' Default: ./Data/AEC/dict.json')

args = parser.parse_args()

glossList = pd.read_json(args.dict_Path)

repeated = []

for glossIndex in glossList:

    word = str.upper(glossList[glossIndex]["gloss"])

    for pos, instance in enumerate(glossList[glossIndex]["instances"]):

        origin = os.sep.join(args.dict_Path.split('/')[0:3])
        target = str(origin) + '/Videos/UNIQUE_NAME/'
        os.makedirs(target, exist_ok = True)

        if args.dict_Path.find('AEC'):
            origin = origin + '/Videos/cropped/'#'/Videos/SEGMENTED_SIGN/'
        else:
            origin = origin + '/Videos/cropped/'

        origin = origin + instance['source_video_name'] + '/' + instance['timestep_vide_name'] + '.mp4'

        #create folder and chec existance
        target = target + instance['source_video_name']
        os.makedirs(target, exist_ok = True)

        target = target + '/' + instance['unique_name'] + '.mp4'
        target = target.encode("UTF-8")

        assert instance['unique_name'] not in repeated
        repeated.append(instance['unique_name'])

        copyfile(origin,target)

        print(target) 
