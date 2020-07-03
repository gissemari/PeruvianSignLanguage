- Set the PYTHONPATH to recognize the utils module (folder):
	export PYTHONPATH=~/Documents/Research/SignLanguage/PeruvianSignLanguaje
- Conversion of time annotation to frame position (or millisecond POS) to read frame
	https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video
- Take into account that there is one FPS for the video input and another FPS for the video output. For example:
	29 -> 29:37 de 28:40
	29.97 -> 28.48 de 28:40