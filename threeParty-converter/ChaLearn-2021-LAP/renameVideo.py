# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:06:21 2021

@author: Joe
"""

import glob
import os

all_files = glob.glob('project/data/mp4/*/*.mp4')


for sample in all_files:

    out_dir = sample.replace('.mp4', '_color.mp4')

    # to avoid adding "_color" two times
    if out_dir.count('_color_color.mp4'):
        out_dir = sample.replace('_color_color.mp4', '_color.mp4')

    os.rename(sample, out_dir)