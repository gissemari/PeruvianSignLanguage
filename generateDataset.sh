# copy utils folder into these directories
cp -r utils 2.Segmentation
cp -r utils 3.Translation/FrameToKeypoint

python 2.Segmentation/segmentPerSRT.py --rawVideoPath ./Data/Videos/RawVideo/ --srtPath ./Data/SRT/SRT_gestures/ --outputVideoPath ./Data/Videos/Segmented_gestures/ --flgGesture 1 

python 3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py --holistic

# errase created files to avoid confusions
rm -r 2.Segmentation/utils
rm -r 3.Translation/FrameToKeypoint/utils

python 4.Preparation/SampleModelFormat.py --words 10 --main_folder_Path ./Data/Keypoints/pkl/Segmented_gestures/ --output_Path ./Data/Dataset/readyToRun/

python 4.Models/Classification.py --timesteps 17 --input_Path ./Data/Dataset/toReshape/ --output_Path ./Data/Dataset/readyToRun/

echo "Press ENTER to exit:"
read ENTER
