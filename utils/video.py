import cv2

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

