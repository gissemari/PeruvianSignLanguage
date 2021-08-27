
#mkdir Logs

python 1.Preprocessing/DatasetXY/Dataset_Preparator.py --words 10 --timesteps 17 --is3D --main_folder_Path ./Data/Keypoints/pkl/Segmented_gestures/ --output_Path ./Data/Dataset/

cp -r utils 4.Models
python 4.Models/Classification.py
rm -r 4.Models/utils

echo "Press ENTER to exit:"
read ENTER
