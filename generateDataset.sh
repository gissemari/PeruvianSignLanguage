# copy utils folder into these directories
cp -r utils 2.Segmentation
cp -r utils 3.Translation/FrameToKeypoint

python 2.Segmentation/segmentPerSRT.py --rawVideoPath ./Data/Videos/RawVideo/ --srtPath ./Data/SRT/SRT_gestures/ --outputVideoPath ./Data/Videos/Segmented_gestures/ --flgGesture 1 

python 3.Translation/FrameToKeypoint/ConvertVideoToDict.py --image --inputPath ./Data/Videos/Segmented_gestures/ --img_output ./Data/Dataset/img/  --dict_output ./Data/Dataset/dict/ --keypoints_output ./Data/Dataset/keypoints/

# errase created files to avoid confusions
rm -r 2.Segmentation/utils
rm -r 3.Translation/FrameToKeypoint/utils

python 4.Preparation/DictToSample.py dict_Path ./Data/Dataset/dict/dict.json --shuffle --leastValue --output_Path ./Data/Dataset/readyToRun/ --words 10

echo "Press ENTER to exit:"
read ENTER
