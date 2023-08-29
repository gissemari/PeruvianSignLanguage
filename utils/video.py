import cv2
import os
import pandas as pd

def createFolder(_path, createFullPath = False):

    if not (os.path.isdir(os.getcwd() + _path) or os.path.isdir(_path)):
        print(f"Directory {_path} has successfully been created")
        if createFullPath:
            os.makedirs(_path, exist_ok=True)
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

def get_list_data(path, _video_ext):
        
    dataset_path = path

    path_list = []

    for folder_path, _ ,video_names in os.walk(dataset_path):
        
        if not video_names:
            continue
        
        for video_name in video_names:

            ext = video_name.split('.')[-1]
            if '_depth' in video_name:
                continue
            if ext in _video_ext:
                video_path = os.sep.join([folder_path, video_name])
                path_list.append(os.path.normpath(video_path))


    return pd.DataFrame({"path":path_list})

def get_list_from_json_dataset(json_path):
    print(json_path)
    print()
    df = pd.read_json(json_path, orient='index')

    data_list = []

    for index, row in df.iterrows():

        gloss = row['gloss']

        for instance in row['instances']:
            dataframe_line = {
                "gloss_id": index,
                "instance_id": instance['instance_id'],
                "label": gloss,
                "timestepNumb": instance['frame_end'],
                "path": instance['source_video_name'],
                "intanceData": instance
            }

            data_list.append(dataframe_line)
    
    return pd.DataFrame(data_list)

def reconstruct_json(df, path):

    reconstructed_data = []

    for index, row in df.iterrows():

        label = row['label']
        instanceData = row['intanceData']

        glossPos = -1

        for indG, gloss in enumerate(reconstructed_data):
            if(gloss["gloss"] == label):
                glossPos = indG

        # in the case word is in the dict
        if glossPos != -1:
            reconstructed_data[glossPos]["instances"].append(row['intanceData'])
        else:
            glossDict = {"gloss": str(label),
                         "instances": [row['intanceData']]}
            reconstructed_data.append(glossDict)

    df = pd.DataFrame(reconstructed_data)
    df.to_json(path, orient='index', indent=2)