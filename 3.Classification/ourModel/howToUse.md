The dataset is available at: https://drive.google.com/drive/u/2/folders/1xWPBmm3qHMl2z4Hz_Iu4m-qBaN4dKb3y

# Steps to prepare dataset - Easy mode

## For AEC or AEC similar format
1. run "./shFiles/generateDataset-AEC.sh"

## For PUCP or external data
1. run "./shFiles/generateDataset-PUCP.sh" 

# Steps to prepare dataset AEC or similars- Advance mode

1. run "python 1.Preprocessing/Video/cropInterpreterBySRT.py"  with this command line options:

  * "--rawVideoPath PATH"   	'PATH' is the directory that points to the group of all the raw videos files
  * "--srtPath PATH"		'PATH' is the directory that points to the group of all the SRT gestures files
  * "--inputName PATH" 
  * "--outputVideoPath PATH"   'PATH' is the directory where you want to save all the Segmented gestures videos (needed for the next step)
  * "--flgGesture #"           '#' is the number of frames per second for the output file
  * "--width #"                 '#' is the width of the SL signer or interpreter that appear in the small rectangle
  * "--height #"                '#' is the height of the SL signer or interpreter that appear in the small rectangle
  * "--x #"                     '#' is the x pixel position that will be the beginning of crop
  * "--y #"                     '#' is the y pixel position that will be the beginning of crop

2. run "python 1.Preprocessing/Video/generateKeypoints.py" with this command line options:

  * "--inputPath PATH"		'PATH' is the directory that points to segmented gestures videos
  * "--img_output PATH"	'PATH' is the directory output where you want to save images that have key point and key lines added in it (use it with '--image')
  * "--keypoints_output PATH"	'PATH' is the directory output where you want to save pkl keypoint data in pkl
  * "--dict_output PATH"	'PATH' is the directory output where you want to save all dataset information (needed in the following command)

3. run "1.Preprocessing/Selector/DictToSample.py"  with this command line options:

  * "--dict_Path PATH"     	'PATH' is the directory that points to the dict that have all the dataset information
  * "--shuffle"		to shuffle instances keys
  * "--leastValue"		to take the word with the least number of instances and cut all word instances to that value 
  * "--wordList _"             '_' is a csv that have a list of word. You can see csv's format in "Data/list.csv"
  * "--output_Path PATH"  	'PATH' is the directory output you want to have the dataset sample formated
  * "--word #" 		'#' is the number of classes

# Steps to prepare dataset for PUCP or external data - Advance mode

1. run "python 1.Preprocessing/Video/cropInterpreterBySRT.py"  with this command line options:

  * "--rawVideoPath PATH"   	'PATH' is the directory that points to the group of all the raw videos files
  * "--srtPath PATH"		'PATH' is the directory that points to the group of all the SRT gestures files
  * "--inputName PATH" 
  * "--outputVideoPath PATH"   'PATH' is the directory where you want to save all the Segmented gestures videos (needed for the next step)
  * "--flgGesture #"           '#' is the number of frames per second for the output file

2. run "python 1.Preprocessing/Video/resizeVideoByFrame.py" with this command line options:

  * "--inputPath PATH"        'Path' is the directory of the input videos to process
  * "--outputPath PATH"        'Path' is the directory where videos will be save

3. run "python 1.Preprocessing/Video/generateKeypoints.py" with this command line options:

  * "--inputPath PATH"		'PATH' is the directory that points to segmented gestures videos
  * "--img_output PATH"	'PATH' is the directory output where you want to save images that have key point and key lines added in it (use it with '--image')
  * "--keypoints_output PATH"	'PATH' is the directory output where you want to save pkl keypoint data in pkl
  * "--dict_output PATH"	'PATH' is the directory output where you want to save all dataset information (needed in the following command)

4. run "1.Preprocessing/Selector/DictToSample.py"  with this command line options:

  * "--dict_Path PATH"     	'PATH' is the directory that points to the dict that have all the dataset information
  * "--shuffle"		to shuffle instances keys
  * "--leastValue"		to take the word with the least number of instances and cut all word instances to that value 
  * "--wordList _"             '_' is a csv that have a list of word. You can see csv's format in "Data/list.csv"
  * "--output_Path PATH"  	'PATH' is the directory output you want to have the dataset sample formated
  * "--word #" 		'#' is the number of classes

# Steps to run the model - Advance mode

1. run "python 3.Classification/ourModel/rnn_classifier.py" with this command line options:

  * "--face"			                  To consider face keypoints 
  * "--hands"			                  To consider both hands keypoints
  * "--pose"			                  To consider pose keypoints
  * "--timesteps #"       	        '#' is the number of timesteps to consider in the dataset
  * "--keypoints_input_Path PATH"   'PATH' is the directory of keypoints inputs  
  * "--keys_input_Path PATH"        'PATH' is the directory of the keys shuffled in the previus command

  * "--plot"                        To activate plot from matplotlib

### To use wandb please add this parameter
  * "--wandb"  		To connect the model with wandb (some files are needed to make wandb work - please read wandb documentation)


## if you are using AEC, just run:
* "python rnn_classifier.py"

# Dataset format
[batch][timestep][feauture]

