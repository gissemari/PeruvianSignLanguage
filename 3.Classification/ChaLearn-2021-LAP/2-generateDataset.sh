echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
python 5.Preparation/DictToSample.py --words $1
echo ""
echo "generating csv in Chalearn..."
sleep 1
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py
python stage2.py

cd ..
cp data/train_val_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos.py
echo "counting frames..."
python count_frames.py
echo "extracting keypoints..."
python extract_keypoint.py
echo "extrancting poseflow..."
sleep 1
python extract_poseflow.py

