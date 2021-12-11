# copy utils folder into these directories
cp -r utils 2.Segmentation
cp -r utils 4.Translation/FrameToKeypoint
cp -r utils 5.Preparation

python 2.Segmentation/segmentPerSRT.py --rawVideoPath ./Data/Videos/RawVideo/ --srtPath ./Data/SRT/SRT_gestures/ --outputVideoPath ./Data/Videos/Segmented_gestures/ --flgGesture 1 

python 4.Translation/FrameToKeypoint/ConvertVideoToDict.py --image --inputPath ./Data/Videos/Segmented_gestures/ --img_output ./Data/Dataset/img/  --dict_output ./Data/Dataset/dict/ --keypoints_output ./Data/Dataset/keypoints/

python 5.Preparation/DictToSample.py --dict_Path ./Data/Dataset/dict/dict.json --shuffle --leastValue --output_Path ./Data/Dataset/readyToRun/ --words 10

# errase created files to avoid confusions
rm -r 2.Segmentation/utils
rm -r 4.Translation/FrameToKeypoint/utils
rm -r 5.Preparation/utils

echo "Press ENTER to exit:"
read ENTER
