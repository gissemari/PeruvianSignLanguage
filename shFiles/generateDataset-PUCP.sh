# copy utils folder into these directories

python 1.Preprocessing/cropInterpreterBySRT.py --rawVideoPath ./Data/PUCP_PSL_DGI156/Videos/original/ --srtPath ./Data/PUCP_PSL_DGI156/SRT/SEGMENTED_SIGN/ --outputVideoPath ./Data/PUCP_PSL_DGI156/Videos/SEGMENTED_SIGN/ --flgGesture 1 

python 1.Preprocessing/Video/ConvertVideoToDict.py --inputPath ./Data/PUCP_PSL_DGI156/Videos/SEGMENTED_SIGN/ --dict_output ./Data/PUCP_PSL_DGI156/ --keypoints_output ./Data/PUCP_PSL_DGI156/Keypoints/

python 1.Preprocessing/Selector/DictToSample.py --dict_Path ./Data/PUCP_PSL_DGI156/dict.json  --output_Path ./Data/PUCP_PSL_DGI156/Selected/ --words 10


echo "Press ENTER to exit:"
read ENTER
