cd ..

#python 1.Preprocessing/Video/cropInterpreterBySRT.py --rawVideoPath ./Data/AEC/Videos/RawVideo/ --srtPath ./Data/AEC/SRT/SRT_SIGN/ --outputVideoPath ./Data/AEC/Videos/SEGMENTED_SIGN/ --flgGesture 1 --x1 380 --y1 988 --width 220 --height 220

python 1.Preprocessing/Video/generateKeypoints.py --inputPath ./Data/AEC/Videos/SEGMENTED_SIGN/  --dict_output ./Data/AEC/ --keypoints_output ./Data/AEC/Keypoints/pkl

python 1.Preprocessing/Selector/DictToSample.py --dict_Path ./Data/AEC/dict.json --shuffle --leastValue --output_Path ./Data/AEC/Selected/ --words 10 --words_File ./Data/list10.csv

echo "Press ENTER to exit:"
read ENTER
