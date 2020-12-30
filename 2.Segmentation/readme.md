This code aims to take the raw video (not yet cropped) and jointly cropp it (to get only the interpreter) and segment by time every gesture annotated in an SRT file (aligned to interpreter).

##### UTILS  #####

- Set the PYTHONPATH to recognize the utils module (folder):
	export PYTHONPATH=~/Documents/Research/SignLanguage/PeruvianSignLanguaje
- Conversion of time annotation (00:00:31,407) to frame position (or millisecond POS) to read frame
	https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video

- Take into account that there is one FPS for the video input and another FPS for the video output. For example:
	29 -> 29:37 de 28:40
	29.97 -> 28.48 de 28:40

- MediaPipe offers already tagging images: https://google.github.io/mediapipe/solutions/holistic

- When using MP4, it gives error but still creates the file. For example:
	OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'mp4 / MP4 (MPEG-4 Part 14)'
	OpenCV: FFMPEG: fallback to use tag 0x7634706d/'mp4v'
	00:01:12.465000 2171 2190

- This is an example on how to call this code:
	python segmentPerSRT.py --rawVideoPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos/RawVideo/ --srtPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/SRT/SRT_gestures/ --inputName ira_alegria --outputVideoPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos/Segmented_gestures/ --flgGesture 1

##### PARAMETERS #####
- rawVideoPath
The path where the original video is located. In our project, it will usually be located at: /PeruvianSignLanguaje/Data/Videos/RawVideo/

- srtPath
The parth where the SRT is located to segment the video assigning frames to signs or sign sentences. They are downloaded from the volunteer's folder. The volunteers where instructed to call SRT files with nivel1 and nivel2 to distinguish between gesture and gesture sentences segmentation. They will usually be located at: PeruvianSignLanguaje/Data/SRT/SRT_gestures/ or PeruvianSignLanguaje/Data/SRT/SRT_gestures_sentences/

- inputName
The name of the file we want to segment. If not introduced, all files in the SRT folder will be processed. We could take also the files inside the rawVideoPath but as we might be preprocessing less number of SRT, we condition on the files here.

- outputVideoPath
The output folder where the segmented gestures will be located. 

- flgGesture
Gesture and sentence gesture need to be distinguish in order to name them in a proper manner. The names of these file follow this format:
	For gesture: WORD_IDX.mp4
	For sentence: IDX.mp4

##### LOGIC #####

- Depending on the inputName, we create a list with all the SRT files in the srtPath or a list of only one element containing the inputName (if specified).
- We iterate over this SRT file list to match the name with the original raw video.
- Videocapture, VideoWriter_fourcc, VideoWriter objects are created to mange all the videos.
- We create a folder with the SRT file name (without the extention) and then create a file for each period of time specified in SRT (segment).



