#source activate chalearn
#export PYTHON_PATH=./PeruvianSignLanguage
echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r utils 5.Preparation
python 5.Preparation/DictToSample.py --dict_Path ../DataReal/AEC/Dataset/dict/dict.json --shuffle --output_Path ../DataReal/AEC/Dataset/readyToRun/ --words_File ../DataReal/list.csv --words 10
echo ""
echo "generating csv in Chalearn..."
cd 3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath ../../../../DataReal/AEC/Dataset/dict/dict.json --keyPath ../../../../DataReal/AEC/Dataset/readyToRun/
python stage2.py 1
cd ..
cp data/train_val_labels_STAGE2.csv project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
python obtain_videos.py --allfiles ../../../DataReal/AEC/Videos/cropped/
echo "counting frames..."
python count_frames.py
echo "extracting keypoints..."
python extract_keypoint.py --src ../../../DataReal/AEC/Dataset/keypoints/ --keyPath ../../../DataReal/AEC/Dataset/readyToRun/
echo "extrancting poseflow..."
python extract_poseflow.py

rm -r 5.Preparation/utils

