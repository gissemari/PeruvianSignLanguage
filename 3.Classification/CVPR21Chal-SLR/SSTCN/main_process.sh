# written by Bin Sun
# email: sun.bi@northeastern.edu

mkdir train_videos
mkdir val_videos
######change path_to_train_videos to your real path for training videos#####################
mv ../../ChaLearn-2021-LAP/project/data/mp4/train/*color* train_videos/
######change path_to_val_videos to your real path for val videos#####################
mv ../../ChaLearn-2021-LAP/project/data/mp4/val/*color* val_videos/

cd data_process
#python wholepose_features_extraction.py --video_path ../train_videos/ --feature_path ../../data-prepare/data/features/train --istrain True
#python wholepose_features_extraction.py --video_path ../val_videos/ --feature_path ../../data-prepare/data/features/val
cd ..
# if you want to delete videos, un common the following command
#rm -rf train_videos
#rm -rf val_videos

####### training #############################
python train_parallel.py --batch_size 40 --dataset_path ../data-prepare/data/features/train/
###### testing ###########################
python test.py
#python test.py --checkpoint_model model_checkpoints/your model
