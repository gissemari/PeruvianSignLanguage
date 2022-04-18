The Peruvian Sign Language Interpretation dataset is shared in this drive link [AEC](https://drive.google.com/file/d/1fwfbheNn_a-HlmWE0lSgofTnmnUupsv3/view?usp=sharing)



# 1.PREPROCESSING #

Every step counts with their own readme file for more details. This readme file is an overview and map of all the different scripts and functions that you can find and do with the data. You can generate the most-heay data with the generateDataset.sh script or find it in this link:

You can also find the conda environment srts2.yml to import all the libraries that use captioning related to SRT files. You can also use the environment.yml to work with all the libraries for videos.


## VIDEO TO ONLY SQUARE (CROP) ##

This code takes the downloaded youtube video from "Aprendo en Casa" and fixes coordinates to crop the section in which the interpreter is performing. This square can later be learned, but for the moment is a fixed space.

Input:
- Raw video downloaded from Youtube 
(./PeruvianSignLanguaje/Data/Videos/RawVideo)

Output:
- Video segmented with fixed coordinates showing only the signer/interpreter
(./PeruvianSignLanguaje/Data/Videos/OnlySquare)

Code:
./PeruvianSignLanguaje/1.Preprocessing/Video/crop_video.py


## RAW TRANSCRIPT TO PER LINE (aligned to audio) ##

Input:
- Transcript all together with no enter between sentences. The text was written by volunteers in simple text format.
(./PeruvianSignLanguaje/Data/Transcripts/all_together)

Output:
- Transcript with every sentence in a different line
(./PeruvianSignLanguaje/Data/Transcripts/per_line)

Code: 
./PeruvianSignLanguaje/1.Preprocessing/PERLINE/convertToLines.py


## PER LINE TO SRT (aligned to audio) ##

Input:
- Transcripts arranged such as every sentence (ended with period, exclamation or question mark) is in one line
(./PeruvianSignLanguaje/Data/Transcripts/per_line)
- SRT downloaded from subtitle.to/ and manually modified to introduce punctuation marks
(./PeruvianSignLanguaje/Data/SRT/SRT_raw)

Output:
- SRT organized by sentence (with time aligned and correct transcript sentence)
(./PeruvianSignLanguaje/Data/SRT/SRT_voice_sentences)

Code:
./PeruvianSignLanguaje/1.Preprocessing/SRTs/convert_subtitle_sentence.py



## SEGMENT GESTURES (aligned to interpreter) ##

Input:
- SRT with annotations in ELAN by sign
- Rawvideo
(./PeruvianSignLanguaje/Data/SRT/SRT_gestures)

Output:
- Segmented already cropped video of interpreter in frames corresponding to each sign
(./PeruvianSignLanguaje/Data/Videos/Segmented_gestures)

Code:
./PeruvianSignLanguaje/2.Segmentation/cropInterpreterBySRT.py (prev: segmentPerSRT.py)


## SEGMENT SIGN SENTENCES (aligned to interpreter##

Input:
- SRT with annotations in ELAN by sign sentence
(./PeruvianSignLanguaje/Data/SRT/SRT_gestures_sentence)

Output:
- Segmented already cropped video of interpreter in frames corresponding to each sign sentence
(./PeruvianSignLanguaje/Data/Videos/Segmented_gesture_sentence)

Code:
./PeruvianSignLanguaje/2.Segmentation/cropInterpreterBySRT.py (prev:segmentPerSRT.py)


## VIDEO TO KEYPOINTS (aligned to interpreter) ##

Input:
- Segmented sign or sign sentence
(./PeruvianSignLanguaje/Data/Videos/Segmented_gestures)

Output:
- Segmented sign or sign sentence images with landmarks
- Pickle files with coordinates of each of the 33 points obtained from mediapipe library (holistic) to annotate landmarks
- Json files with coordinates of the keypoint landmarks
(./PeruvianSignLanguaje/Data/Keypoints)

Code:
./PeruvianSignLanguaje/3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py

Parameters:
--image creates pickle files of images. However it is recommended not to use it
--input_path


# 2.SEGMENTATION #

Here we will place all the models to perform automatic segmentations (identification of sign units or sign sentence units).




# 3.CLASSIFICATION #

To run the ChaLearn Model go to ./PeruvianSignLanguaje/3.Classification
It works with a different data folder where it creates only the classes that the model is working with.



# 4.TRANSLATION #

Here we will place all the models to perform the end-to-end translation.



#### PIPELINE USED FOR LREC ####
For the AEC dataset:
1. cropInterpreterBySRT.py
2. convertVideoToDict.py
3. Classification\ChaLearn

# REPOSITORIES USED FOR THE CLASSIFICATION

```
@InProceedings{De_Coster_2021_CVPR,
    author    = {De Coster, Mathieu and Van Herreweghe, Mieke and Dambre, Joni},
    title     = {Isolated Sign Recognition From RGB Video Using Pose Flow and Self-Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {3441-3450}
}

@InProceedings{jiang2021sign,
    author    = {Jiang, Songyao and Sun, Bin and Wang, Lichen and Bai, Yue and Li, Kunpeng and Fu, Yun},
    title     = {Sign Language Recognition via Skeleton-Aware Multi-Model Ensemble},
    journal   = {arXiv preprint arXiv:2110.06161},
    year      = {2021}
}
```
