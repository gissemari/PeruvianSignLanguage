echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..

python 1.Preprocessing/Selector/DictToSample.py --dict_Path ./Data/AEC/dict.json --shuffle --output_Path ./Data/AEC/Selected/ --words_File ./Data/list.csv --words $1
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ./../../../Data/AEC/dict.json --keyPath ./../../../Data/AEC/Selected/
python stage2.py 1

cd ..
cp data/train_val_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py --allfiles ./../../Data/AEC/Videos/cropped/
echo "counting frames..."
python count_frames.py
echo "extracting keypoints..."
python extract_keypoint.py --src ./../../Data/AEC/Keypoints/pkl/ --keyPath ./../../Data/AEC/Selected/
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py

