# copy utils folder into these directories
cp -r utils 2.Segmentation
cp -r utils 3.Translation/FrameToKeypoint

python 2.Segmentation/segmentPerSRT.py --rawVideoPath ./Data/Videos/RawVideo/ --srtPath ./Data/SRT/SRT_gestures/ --outputVideoPath ./Data/Videos/Segmented_gestures/ --flgGesture 1

python 3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py --holistic

# errase created files to avoid confusions
rm -r 2.Segmentation/utils
rm -r 3.Translation/FrameToKeypoint/utils

python 1.Preprocessing/DatasetXY/Dataset_Preparator.py --words 20 --timesteps 40 --is3D --main_folder_Path ./Data/Keypoints/pkl/Segmented_gestures/ --output_Path ./Data/Dataset/

echo "Press ENTER to exit:"
read ENTER