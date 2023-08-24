Libraries to use:
- pip install opencv-python

To segment a video, first we have to calculate the number of frames per second. We added this logic inside crop_video.py
- python video_fps.py --inputName ira_alegria

Then we have to start to crop, assigning, for the moment, fixed coordinates:
-  python crop_video.py --inputName ira_alegria

If output is not produced due to errors such as demultiplexer, you need to check the sizes of the crop. Make sure you use height, width, as opposite as the width, height in the definition of the VideoWriter.

SRT has a format of millisecond (1-1000) but the string format of a timestamp is in microsecond (10^-6)