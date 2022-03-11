#source activate chalearn
#export PYTHON_PATH=./PeruvianSignLanguage
echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r utils 5.Preparation
python 5.Preparation/DictToSample.py --dict_Path Data/AEC/Dataset/dict/dict.json --shuffle --output_Path Data/AEC/Dataset/readyToRun/ --words_File Data/list10.csv --words 10
echo ""
echo "generating csv in Chalearn..."
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ../../../Data/AEC/Dataset/dict/dict.json --keyPath ../../../Data/AEC/Dataset/readyToRun/
python stage2.py 1
cd ..
cp data/train_val_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
python obtain_videos.py --allfiles ../../Data/AEC/Videos/SEGMENTED_SIGN/
echo "counting frames..."
python count_frames.py
echo "extracting keypoints..."
python extract_keypoint.py --src ../../Data/AEC/Dataset/keypoints/ --keyPath ../../Data/AEC/Dataset/readyToRun/
echo "extrancting poseflow..."
python extract_poseflow.py

rm -r 5.Preparation/utils

