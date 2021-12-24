source activate chalearn
export PYTHON_PATH=/home/bejaranog/signLanguage/PeruvianSignLanguage
echo ""
echo "generating data in Peruvian Sign language project..."
sleep 1
cd ../..
cp -r /home/bejaranog/signLanguage/PeruvianSignLanguage/utils /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation
python /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation/DictToSample.py --dict_Path /data/bejaranog/signLanguage/Data/AEC/Dataset/dict/dict.json --shuffle --output_Path /data/bejaranog/signLanguage/Data/AEC/Dataset/readyToRun/ --words_File /home/bejaranog/signLanguage/PeruvianSignLanguage/Data/list.csv --words 10
echo ""
echo "generating csv in Chalearn..."
cd /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/data
python excel.py --dictPath /data/bejaranog/signLanguage/Data/AEC/Dataset/dict/dict.json --keyPath /data/bejaranog/signLanguage/Data/AEC/Dataset/readyToRun/
python stage2.py 1
cd ..
cp /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/data/train_val_labels_STAGE2.csv /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/project
echo ""
echo "train_val_labels_STAGE2.csv copied in project file"
echo ""
echo "obtaining videos from Peruvian Sign language project..."
python /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/obtain_videos.py --allfiles /data/bejaranog/signLanguage/Data/AEC/Videos/cropped/
echo "counting frames..."
python /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/count_frames.py
echo "extracting keypoints..."
python /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/extract_keypoint.py --src /data/bejaranog/signLanguage/Data/AEC/Dataset/keypoints/ --keyPath /data/bejaranog/signLanguage/Data/AEC/Dataset/readyToRun/
echo "extrancting poseflow..."
python /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/extract_poseflow.py

rm -r /home/bejaranog/signLanguage/PeruvianSignLanguage/5.Preparation/utils

