# Steps to prepare dataset and Run the model - Easy mode

1. run "./generateDataset.sh" 
2. run "./rnn_classifier.sh"

# Steps to prepare dataset and Run the model - Advance mode

1. run "python 2.Segmentation/segmentPerSRT.py"  with this command line options:

  * "--rawVideoPath PATH"   	'PATH' is the directory that points to the group of all the raw videos files
  * "--srtPath PATH"		'PATH' is the directory that points to the group of all the SRT gestures files
  * "--inputName PATH" 
  * "--outputVideoPath PATH"   'PATH' is the directory where you want to save all the Segmented gestures videos (needed for the next step)
  * "--flgGesture #"           '#' is the number of frames per second for the output file

2. run "python 3.Translation/FrameToKeypoint/ConvertVideoToDict.py" with this command line options:

  * "--image"
  * "--inputPath PATH"		'PATH' is the directory that points to segmented gestures videos
  * "--img_output PATH"	'PATH' is the directory output where you want to save images that have key point and key lines added in it (use it with '--image')
  * "--keypoints_output PATH"	'PATH' is the directory output where you want to save pkl keypoint data in pkl
  * "--dict_output PATH"	'PATH' is the directory output where you want to save all dataset information (needed in the following command)


3. run "python 4.Preparation/DictToSample.py"  with this command line options:


  * "--dict_Path PATH"     	'PATH' is the directory that points to the dict that have all the dataset information
  * "--shuffle"		to shuffle instances keys
  * "--leastValue"		to take the word with the least number of instances and cut all word instances to that value 
  * "--wordList _"             '_' is a list of word. This has to be written like this example "--wordList WORD1 WORD2 WORD3" 
  * "--output_Path PATH"  	'PATH' is the directory output you want to have the dataset sample formated
  * "--word #" 		'#' is the number of classes

4. run "python 4.Models/rnn_classifier.py" with this command line options:

  * "--face"			To consider face keypoints 
  * "--hands"			To consider both hands keypoints
  * "--pose"			To consider pose keypoints
  * "--timesteps #"       	'#' is the number of timesteps to consider in the dataset
  * "--wandb"  		To connect the model with wandb (some files are needed to make wandb work - please read wandb documentation)
  * "--keypoints_input_Path PATH" 'PATH' is the directory of keypoints inputs  
  * "--keys_input_Path PATH"   'PATH' is the directory of the keys shuffled in the previus command (SampleModelFormat.py) 

# Dataset format

[batch][timestep][feauture]

# Troubleshooting

If you are using terminal, it is possible to have some problems with local libraries imports
use "cp -r utils PATH" to make a local library copy where it is needed (PATH)
*then you can use rm -r PATH/utils (to remove the local library copy)*
