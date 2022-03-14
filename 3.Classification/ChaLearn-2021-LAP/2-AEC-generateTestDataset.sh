echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..

python 1.Preprocessing/Selector/DictToSample.py --dict_Path ./Data/AEC/dict.json --shuffle --output_Path ./Data/AEC/Selected/ --words_File ./Data/list.csv --words $1
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ./../../../Data/AEC/dict.json --keyPath ./../../../Data/AEC/Selected/ --splitRate 1.0
python stage2.py --train 0

cd ..
cp data/test_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --allfiles ./../../Data/AEC/Videos/SEGMENTED_SIGN/ --train 0
echo "counting frames..."
python count_frames.py --train 0
echo "extracting keypoints..."
python extract_keypoint.py --src ./../../Data/AEC/Keypoints/pkl/ --keyPath ./../../Data/AEC/Selected/ --train 0
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py --train 0
