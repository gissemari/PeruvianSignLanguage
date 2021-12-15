echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r utils 5.Preparation
python 5.Preparation/DictToSample.py --dict_Path ./Data/PUCP_PSL_DGI156/Dataset/dict/dict.json --shuffle --output_Path ./Data/PUCP_PSL_DGI156/Dataset/readyToRun/ --words $1
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ./../../../Data/PUCP_PSL_DGI156/Dataset/dict/dict.json --keyPath ./../../../Data/PUCP_PSL_DGI156/Dataset/readyToRun/
python stage2.py

cd ..
cp data/train_val_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --allfiles ./../../Data/PUCP_PSL_DGI156/Videos/SEGMENTED_SIGN/
echo "counting frames..."
python count_frames.py
echo "extracting keypoints..."
python extract_keypoint.py --src ./../.././Data/PUCP_PSL_DGI156/Dataset/keypoints/ --keyPath ./../../../Data/PUCP_PSL_DGI156/Dataset/readyToRun/
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py

rm -r ../../5.Preparation/utils

