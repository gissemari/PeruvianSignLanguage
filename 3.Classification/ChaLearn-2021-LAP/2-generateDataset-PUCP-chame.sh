echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r utils 5.Preparation
python 5.Preparation/DictToSample.py --dict_Path ../DataReal/PUCP_PSL_DGI156/Dataset/dict/dict.json --shuffle --output_Path ../DataReal/PUCP_PSL_DGI156/Dataset/readyToRun/ --words_File ../DataReal/list.csv --words 10
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ../../../../DataReal/PUCP_PSL_DGI156/Dataset/dict/dict.json --keyPath ../../../../DataReal/PUCP_PSL_DGI156/Dataset/readyToRun/ --splitRate 1.0
python stage2.py --train 0

cd ..
cp data/train_val_labels_STAGE2.csv project
cp data/test_labels_STAGE2.csv project

echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --train 0 --allfiles ../../../DataReal/PUCP_PSL_DGI156/Videos/cropped/
echo "counting frames..."
python count_frames.py --train 0
echo "extracting keypoints..."
python extract_keypoint.py --train 0 --src ../../../DataReal/PUCP_PSL_DGI156/Dataset/keypoints/ --keyPath ../../../DataReal/PUCP_PSL_DGI156/Dataset/readyToRun/
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py --train 0

