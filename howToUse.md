# Steps to prepare dataset and Run the model - Easy mode

1. run "./generateDataset.sh" 
2. run "./runClassifier.sh"

# Steps to prepare dataset and Run the model - Advance mode

1. run "python 2.Segmentation/segmentPerSRT.py"  with this command line options:

  * "--rawVideoPath PATH"   	'PATH' is the directory that points to the group of all the raw videos files
  * "--srtPath PATH"		'PATH' is the directory that points to the group of all the SRT gestures files
  * "--outputVideoPath PATH"   'PATH' is the directory where you want to save all the Segmented gestures videos (needed for the next step)
  * "--flgGesture #"           '#' is the number of frames per second for the output file

2. run "python 3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py" with this command line options:

  * "--holistic" 		To use Mediapipe holistic model
  * "--Pose"			To use Mediapipe Pose model
  * "--hands"			To use Mediapipe hands model
  * "--face_mesh"		To use Mediapipe Face model
  
  *If you use --holistic, use it without (pose, hands and face mesh)*
  
  * "--inputPath PATH"		'PATH' is the directory that points to segmented gestures videos
  * "--img_output PATH"	'PATH' is the directory output where you want to save images that have key point and key lines added in it
  * "--pkl_output PATH"	'PATH' is the directory output where you want to save pkl key point data in pkl (needed for the next step)
  * "--json_output PATH"	'PATH' is the directory output where you want to save pkl key point data in json

3. run "python 4.Preparation/KeypointsFramesToSample.py"  with this command line options:

  * "--word #" 		'#' is the number of classes
  * "--main_folder_Path PATH"  'PATH' is the directory that points to the file that groups all segmented gesture keypoints (pkl)
  * "--output_Path PATH"       'PATH' is the directory output you want to have the dataset samples (without format)

4. run "python 4.Preparation/SampleModelFormat.py"  with this command line options:

  * "--timesteps #"       '#' is the number of timesteps 
  * "input_Path PATH"     'PATH' is the directory that points to the dataset sample file
  * "--output_Path PATH"  'PATH' is the directory output you want to have the dataset sample

5. "python 4.Models/Classification.py" with this command line options:

  * "--wandb"  		To connect the model with wandb (some files are needed to make wandb work - please read wandb documentation)

# Dataset format

[batch][timestep][feauture]

# Troubleshooting

If you are using terminal, it is possible to have some problems with local libraries imports
use "cp -r utils PATH" to make a local library copy where it is needed (PATH)
*then you can use rm -r PATH/utils (to remove the local library copy)*
