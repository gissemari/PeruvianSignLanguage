import cv2
import os

def createFolder(_path, createFullPath = False):

	if not os.path.isdir(os.getcwd() + _path):
		print(f"Directory {_path} has successfully been created")
		if createFullPath:
			os.makedirs(_path)
		else:
			os.mkdir(_path)
		return 1
	return 0

def getNumFrames(cap):
	# Read number of frames and version
	(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

	if int(major_ver) < 3:
		fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
		print(
	        "Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0} {1} {2}".format(fps, cap.get(cv2.CAP_PROP_POS_MSEC), cap.get(cv2.CAP_PROP_FRAME_COUNT) ))
	else:
		fps = cap.get(cv2.CAP_PROP_FPS)
		print(
	        "Frames per second using video.get(cv2.CAP_PROP_FPS) : {0} {1} {2}".format(fps, cap.get(cv2.CAP_PROP_POS_MSEC) , cap.get(cv2.CAP_PROP_FRAME_COUNT) ))

	return fps

