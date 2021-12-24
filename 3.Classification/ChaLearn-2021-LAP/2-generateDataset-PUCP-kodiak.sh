echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r utils 5.Preparation
python 5.Preparation/DictToSample.py --dict_Path /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/dict/dict.json --shuffle --output_Path /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/readyToRun/ --words_File ./Data/list.csv --words $1
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/dict/dict.json --keyPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/readyToRun/ --splitRate 1.0
python stage2.py --train 0

cd ..
cp /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/data/train_val_labels_STAGE2.csv /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/project
cp /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/data/test_labels_STAGE2.csv project

echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --train 0 --allfiles /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Videos/cropped/
echo "counting frames..."
python count_frames.py --train 0
echo "extracting keypoints..."
python extract_keypoint.py --train 0 --src /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/keypoints/ --keyPath /data/bejaranog/signLanguage/Data/PUCP_PSL_DGI156/Dataset/readyToRun/
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py --train 0

