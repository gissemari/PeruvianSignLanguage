# Steps to prepare dataset and Run the model - Easy mode

1. run "./generateDataset.sh" 
2. run "./runClassifier.sh"

# Steps to prepare dataset and Run the model - Advance mode

1. run "python 2.Segmentation/segmentPerSRT.py"  with this command line options:

  * "--rawVideoPath PATH"   	'PATH' is the directory that points to the file that groups all the raw videos
  * "--srtPath PATH"		'PATH' is the directory that points to the file that groups all the SRT gestures
  * "--outputVideoPath PATH"   'PATH' is the directory where you want to save all the Segmented gestures videos
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

3. run "python 1.Preprocessing/DatasetXY/Dataset_Preparator.py"  with this command line options:

  * "--word #" 		'#' is the number of classes
  * "--timesteps #"   		'#' is the number of timesteps
  * "--is3D"			To have 3 dimention data [batch][timestep][feauture]
  * "--main_folder_Path PATH"  'PATH' is the directory that points to the file that groups all segmented gesture keypoints (pkl)
  * "--output_Path PATH"       'PATH' is the directory output you want to have the dataset

4. "python 4.Models/Classification.py" with this command line options:

  * "--wandb"  		To connect the model with wandb (some files are needed to make wandb work - please read wandb documentation)

# Troubleshooting

If you are using terminal, it is possible to have some problems with local libraries imports
use "cp -r utils PATH" to make a local library copy where it is needed (PATH)
*then you can use rm -r PATH/utils (to remove the local library copy)*
