source activate chalearn
# copy utils folder into these directories
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation

python /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation/segmentPerSRT.py --rawVideoPath /data/bejaranog/signLanguage/Data/AEC/Videos/RawVideo/ --srtPath /data/bejaranog/signLanguage/Data/AEC/SRT/SRT_SIGN/ --outputVideoPath /data/bejaranog/signLanguage/Data/AEC/Videos/SEGMENTED_SIGN/ --flgGesture 1 

python /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/multiprocess.py --inputPath /data/bejaranog/signLanguage/Data/AEC/Videos/SEGMENTED_SIGN/ --dict_output /data/bejaranog/signLanguage/Data/AEC/Dataset/dict/ --keypoints_output /data/bejaranog/signLanguage/Data/AEC/Dataset/keypoints/

#python /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/ConvertVideoToDict.py --inputPath /data/bejaranog/signLanguage/Data/AEC/Videos/SEGMENTED_SIGN/ --img_output  /data/bejaranog/signLanguage/Data/AEC/Dataset/img/  --dict_output /data/bejaranog/signLanguage/Data/AEC/Dataset/dict/ --keypoints_output /data/bejaranog/signLanguage/Data/AEC/Dataset/keypoints/

#python 5.Preparation/DictToSample.py --dict_Path ./Data/Dataset/dict/dict.json --shuffle --leastValue --output_Path ./Data/Dataset/readyToRun/ --words 10

# errase created files to avoid confusions
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation/utils
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/utils
#rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation/utils

echo "Press ENTER to exit:"
read ENTER
