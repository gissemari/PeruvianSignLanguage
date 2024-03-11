In order to tag each frame of a video with the keypoints landmark, we make use of the library mediapipe. This library offers 4 sets of keypoints: face (468), hands (21 x 2), pose (25) and holistic (that includes all the previos one and additional pose points: 468 + 42 + 33). We will workw ith only pose for the moment but try later more keypoints

ConvertToDict has all the logic now, but we are maintaining convertToKeyPoints because it has part of the code that we can reuse.

##### UTILS #####
- You can install the mediapipe library with: pip install mediapipe or use the conda environment file already in the root of the project srts2.yml
- The exact identificator of keypoints landmarks are defined in: https://google.github.io/mediapipe/solutions/pose.html
- The results of process(imageBGR) are as follows:
	holisResults._fields > ('pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks', 'face_landmarks')
- How to run?
	python ConvertVideoToKeypoint.py --img_input ./Data/Videos/Segmented_gestures/ira_alegria

##### PARAMETERS #####
