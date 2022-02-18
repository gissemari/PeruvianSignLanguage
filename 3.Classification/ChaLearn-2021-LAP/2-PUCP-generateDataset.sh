echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..

python 1.Preprocessing/Selector/DictToSample.py --dict_Path ./Data/PUCP_PSL_DGI156/dict.json --shuffle --output_Path ./Data/PUCP_PSL_DGI156/Selected/ --words $1 --words_File ./Data/list10.csv
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ../../../Data/PUCP_PSL_DGI156/dict.json --keyPath ../../../Data/PUCP_PSL_DGI156/Selected/ --splitRate 1.0
python stage2.py --train 0

cd ..
cp data/train_val_labels_STAGE2.csv project
cp data/test_labels_STAGE2.csv project

echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --train 0 --allfiles ../../Data/PUCP_PSL_DGI156/Videos/cropped/
echo "counting frames..."
python count_frames.py --train 0
echo "extracting keypoints..."
python extract_keypoint.py --train 0 --src ../../Data/PUCP_PSL_DGI156/Keypoints/pkl/ --keyPath ../../Data/PUCP_PSL_DGI156/Selected/
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py --train 0
