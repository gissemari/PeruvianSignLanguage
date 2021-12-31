source activate chalearn
# copy utils folder into these directories
PATH_ASL="/home/cc/ASL/PeruvianSignLanguage"
cp -r $PATH_ASL/utils $PATH_ASL/2.Segmentation
cp -r $PATH_ASL/utils $PATH_ASL/4.Translation/FrameToKeypoint
cp -r $PATH_ASL/utils $PATH_ASL/5.Preparation

#python 2.Segmentation/segmentPerSRT.py --rawVideoPath $PATH_ASL/Data/AEC/Videos/RawVideo/ --srtPath $PATH_ASL/Data/AEC/SRT/SRT_SIGN/ --outputVideoPath $PATH_ASL/Data/AEC/Videos/SEGMENTED_SIGN/ --flgGesture 1 --width 220 --height 220 --x1 380 --y1 988

python 4.Translation/FrameToKeypoint/ConvertVideoToDict.py --inputPath $PATH_ASL/Data/AEC/Videos/SEGMENTED_SIGN/ --dict_output $PATH_ASL/Data/AEC/Dataset/dict/ --keypoints_output $PATH_ASL/Data/AEC/Dataset/keypoints/

# ConvertVideoToDict was replaced by multiprocess
#python 4.Translation/FrameToKeypoint/ConvertVideoToDict.py --inputPath $PATH_ASL/Data/AEC/Videos/SEGMENTED_SIGN/ --img_output  $PATH_ASL/Data/AEC/Dataset/img/  --dict_output $PATH_ASL/Data/AEC/Dataset/dict/ --keypoints_output $PATH_ASL/Data/AEC/Dataset/keypoints/

#python 5.Preparation/DictToSample.py --dict_Path ./Data/Dataset/dict/dict.json --shuffle --leastValue --output_Path ./Data/Dataset/readyToRun/ --words 10

# errase created files to avoid confusions
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation/utils
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/utils
#rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation/utils

echo "Press ENTER to exit:"
read ENTER
