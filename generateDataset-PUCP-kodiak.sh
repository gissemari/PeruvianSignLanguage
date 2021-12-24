source activate chalearn
# copy utils folder into these directories
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation

python /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation/segmentPerSRT.py --rawVideoPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Videos/original/ --srtPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/SRT/SRT_SEGMENTED_SIGN_SUB3/ --outputVideoPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Videos/SEGMENTED_SIGN/ --flgGesture 1 

python /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/ConvertVideoToDict.py --inputPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Videos/SEGMENTED_SIGN/ --dict_output /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/dict/ --keypoints_output /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/keypoints/

#python 5.Preparation/DictToSample.py --dict_Path ./Data/PUCP_PSL_DGI156/Dataset/dict/dict.json  --output_Path ./Data/PUCP_PSL_DGI156/Dataset/readyToRun/ --words 10

# errase created files to avoid confusions
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/2.Segmentation/utils
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/4.Translation/FrameToKeypoint/utils
rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation/utils

echo "Press ENTER to exit:"
read ENTER
