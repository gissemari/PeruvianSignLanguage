
cd data
python excel_new.py
python stage2.py 1

cd ..
#cp data/train_val_labels_STAGE2.csv project/
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
sleep 1
python obtain_videos_new.py
echo "counting frames..."
python count_frames_new.py
#python count_frames.py --train 1
echo "extracting keypoints..."
python extract_keypoint_new.py 
echo "extrancting poseflow..."
sleep 1
python extract_poseflow_new.py

