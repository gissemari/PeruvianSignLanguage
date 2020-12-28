In order to tag each frame of a video with the keypoints landmark, we make use of the library mediapipe.
pip install mediapipe
The exact identificator of keypoints: https://google.github.io/mediapipe/solutions/pose.html

holisResults._fields
('pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks', 'face_landmarks')

- How to run?
	python ConvertVideoToKeypoint.py --img_input /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos/Segmented_gestures/ira_alegria