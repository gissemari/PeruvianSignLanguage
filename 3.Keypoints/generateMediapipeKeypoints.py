# Standard library imports
import argparse
import warnings
import os
import sys

# Third party imports
import cv2
import pandas as pd
import numpy as np
import h5py
import mediapipe as mp
import pickle as pkl

# Local imports
sys.path.append('../../')
import utils.video as uv
import utils.mediapipe_functions as mpf

# Title
parser = argparse.ArgumentParser(description='Use of Holistic Mediapipe model to generate a Dict')

# File paths
parser.add_argument('--inputPath', type=str, default="../../Data/AEC/Videos/SEGMENTED_SIGN/",
                    help='relative path of images input.' + ' Default: ./Data/AEC/Videos/SEGMENTED_SIGN/')
parser.add_argument('--dict_output', type=str, default="../../Data/AEC/",
                    help='relative path of scv output set of landmarks.' +' Default: ./Data/Dataset/dict/')
parser.add_argument('--keypoints_output', type=str, default="../../Data/AEC/AEC_mediapipe.hdf5",
                    help='relative path of csv output set of landmarks.' + ' Default: ./Data/Dataset/keypoints/')
parser.add_argument('--new_flow', type=bool, default=False)

args = parser.parse_args()

if args.new_flow:

    args.inputPath = os.path.normpath(args.inputPath)
    args.dict_output = os.path.normpath(os.sep.join([args.dict_output,"dict.json"]))

    df_video_paths = uv.get_list_data(args.inputPath, ['mp4', 'mov'])
    assert 1 == 2
    holistic = mpf.model_init()

    h5_file = h5py.File(args.keypoints_output, 'w')
    LSP = []

    video_errors = []

    for _num, videoPath in enumerate(df_video_paths['path']):
        print(videoPath)

        cap = cv2.VideoCapture(videoPath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (cap.isOpened() is False):
            print("Unable to read camera feed", videoPath)
            video_errors.append(videoPath)
            continue

        # Read video and collect results
        ret, frame = cap.read()
        results = []
        while ret is True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.warn("deprecated", DeprecationWarning)
                frame_kp = mpf.frame_process(holistic, frame)
                results.append(frame_kp)

            ret, frame = cap.read()

        results = np.array(results)
        # get the file name -> split the ext -> split by "_" -> upper
        label = os.path.splitext(os.path.basename(videoPath))[0].split('_')[0].upper()
        unique_name = os.sep.join(videoPath.split(os.sep)[-2:])
        
        # accumulate data
        grupo_name = f"{_num}"
        h5_file.create_group(grupo_name)
    
        h5_file[grupo_name]['video_name'] = unique_name
        h5_file[grupo_name]['label'] = label
        h5_file[grupo_name]['data'] = results

        print(f"Video processed: {_num}")

        glossInst = {
            "image_dimention": {
                "height": frame_height,
                "witdh": frame_width
            },
            #"keypoints_iD": f"{num}",
            #"image_path": pklImagePath,
            "frame_end": results.shape[0],
            "frame_start": 1,
            "instance_id": _num,
            "signer_id": -1,
            "unique_name": unique_name,
            "source": "LSP",
            "split": "",
            "variation_id": -1,
            "source_video_name": os.sep.join(videoPath.split(os.sep)[-2:-1]),
            "timestep_vide_name": os.path.splitext(os.path.basename(videoPath))[0]
        }

        # check if there is a gloss asigned with "word"
        glossPos = -1

        for indG, gloss in enumerate(LSP):
            if(gloss["gloss"] == label):
                glossPos = indG

        # in the case word is in the dict
        if glossPos != -1:
            LSP[glossPos]["instances"].append(glossInst)
        else:
            glossDict = {"gloss": str(label),
                        "instances": [glossInst]}
            LSP.append(glossDict)

        df = pd.DataFrame(LSP)
        df.to_json(args.dict_output, orient='index', indent=2)

    h5_file.close()
    mpf.close_model(holistic)

# Old flow
else:

    #########################
    # MODELS(Mediapipe)
    #
    # -Holistic
    ##############

    print("\n#####\nHolistic Model\n#####\n")
    mp_holistic = mp.solutions.holistic

    #########################
    # MODELS PARAMETERS
    ##############

    # HOLISTIC parameters.
    holistic = mp_holistic.Holistic(static_image_mode=True,
                                    model_complexity=2,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    ####
    # Check if route is a folder. If so, create results for each of the videos
    # If no, create results for only the video passed
    #
    ###

    # Folder list of videos's frames

    if os.path.isdir(os.getcwd()+'/'+args.inputPath) or os.path.isdir(args.inputPath):
        folder_list = [file for file in (os.listdir(os.getcwd()+'/'+args.inputPath))
                    if (os.path.isdir(os.getcwd()+'/'+args.inputPath+file) or os.path.isdir(args.inputPath+file))]
        print("InputPath is Directory\n")
    else:
        folder_list = [args.inputPath]

        print("Folder List:\n")
        print(folder_list)

    #print(args.keypoints_output)
    #uv.createFolder(args.img_output)
    uv.createFolder(args.dict_output)
    uv.createFolder(args.keypoints_output) #,createFullPath=True)

    IdCount = 1
    LSP = []
    video_errors = []

    dictPath = args.dict_output+'/'+"dict"+'.json'

    # Iterate over the folders of each video in Video/Segmented_SIGN
    for videoFolderName in folder_list:

        videoFolderPath = args.inputPath

        videoFolderList = [file for file in os.listdir(os.sep.join([os.getcwd(),videoFolderPath + videoFolderName]))]

        cropVideoPath = videoFolderPath + videoFolderName
        uv.createFolder(cropVideoPath, createFullPath=True)
        uv.createFolder(args.keypoints_output+'/'+videoFolderName, createFullPath=True)

        for videoFile in videoFolderList:
            
            word = videoFile.split("_")[0]
            #if word not in ["G-R", "bien", "comer", "cuánto", "dos", "porcentaje", "proteína", "sí", "tú", "yo"]:
            #    continue

            keypointsDict = []

            videoSegFolderName = videoFolderPath+videoFolderName+'/'+videoFile
            videoSegFolderName = videoSegFolderName.split('/')
            videoSegFolderName = os.sep.join([*videoSegFolderName])

            pklInitPath = os.sep.join([*args.keypoints_output.split('/')])

            pklKeypointsCompletePath = os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl'])

            pklKeypointsPath = os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl'])
            print(videoSegFolderName)
            # Create a VideoCapture object
            cap = cv2.VideoCapture(os.getcwd()+'/'+videoSegFolderName)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Check if camera opened successfully
            if (cap.isOpened() is False):
                print("Unable to read camera feed", os.getcwd()+videoSegFolderName)
                video_errors.append(videoFolderPath+'/'+videoFile)
                continue

            uniqueVideoName = word + '_' + str(IdCount)
            #video = cv2.VideoWriter(cropVideoPath + word + '_' + str(IdCount)+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))
            #video = cv2.VideoWriter(cropVideoPath+'/' + uniqueVideoName +'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,(220,220))

            idx = 0

            w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Init video size:",w_frame, h_frame)
            ret, frame = cap.read()

            # While a frame was read
            while ret is True:

                idx += 1  # Frame count starts at 1

                # temporal variables
                kpDict = {}

                # Convert the BGR image to RGB before processing.
                imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = imageBGR.shape

                holisResults = holistic.process(imageBGR)

                # POSE

                kpDict["pose"]={}
                if holisResults.pose_landmarks:

                    kpDict["pose"]["x"] = [point.x for point in holisResults.pose_landmarks.landmark]
                    kpDict["pose"]["y"] = [point.y for point in holisResults.pose_landmarks.landmark]

                else:
                    kpDict["pose"]["x"] = [1.0 for point in range(0, 33)]
                    kpDict["pose"]["y"] = [1.0 for point in range(0, 33)]

                # HANDS

                # Left hand
                kpDict["left_hand"]={}
                if(holisResults.left_hand_landmarks):

                    kpDict["left_hand"]["x"] = [point.x for point in holisResults.left_hand_landmarks.landmark]
                    kpDict["left_hand"]["y"] = [point.y for point in holisResults.left_hand_landmarks.landmark]

                else:
                    kpDict["left_hand"]["x"] = [1.0 for point in range(0, 21)]
                    kpDict["left_hand"]["y"] = [1.0 for point in range(0, 21)]

                # Right hand
                kpDict["right_hand"]={}
                if(holisResults.right_hand_landmarks):

                    kpDict["right_hand"]["x"] = [point.x for point in holisResults.right_hand_landmarks.landmark]
                    kpDict["right_hand"]["y"] = [point.y for point in holisResults.right_hand_landmarks.landmark]

                else:
                    kpDict["right_hand"]["x"] = [1.0 for point in range(0, 21)]
                    kpDict["right_hand"]["y"] = [1.0 for point in range(0, 21)]

                # Face mesh

                kpDict["face"]={}

                if(holisResults.face_landmarks):
                    '''
                    nose_points = [1,5,6,218,438]
                    mouth_points = [78,191,80,81,82,13,312,311,310,415,308,
                                    95,88,178,87,14,317,402,318,324,
                                    61,185,40,39,37,0,267,269,270,409,291,
                                    146,91,181,84,17,314,405,321,375]
                    #mouth_points = [0,37,39,40,61,185,267,269,270,291,409, 
                    #                12,38,41,42,62,183,268,271,272,292,407,
                    #                15,86,89,96,179,316,319,325,403,
                    #                17,84,91,146,181,314,321,375,405]
                    left_eyes_points = [33,133,157,158,159,160,161,173,246,
                                        7,144,145,153,154,155,163]
                    left_eyebrow_points = [63,66,70,105,107]
                                        #46,52,53,55,65]
                    right_eyes_points = [263,362,384,385,386,387,388,398,466,
                                        249,373,374,380,381,382,390]
                    right_eyebrow_points = [293,296,300,334,336]
                                            #276,282,283,285,295]

                    #There are 97 points
                    exclusivePoints = nose_points
                    exclusivePoints = exclusivePoints + mouth_points
                    exclusivePoints = exclusivePoints + left_eyes_points
                    exclusivePoints = exclusivePoints + left_eyebrow_points
                    exclusivePoints = exclusivePoints + right_eyes_points
                    exclusivePoints = exclusivePoints + right_eyebrow_points
                    '''

                    kpDict["face"]["x"] = [point.x for point in holisResults.face_landmarks.landmark]
                    kpDict["face"]["y"] = [point.y for point in holisResults.face_landmarks.landmark]

                    '''
                    for posi, data_point in enumerate(holisResults.face_landmarks.landmark):
                        if posi in exclusivePoints:
                            list_X.append(data_point.x)
                            list_Y.append(data_point.y)
                    '''
                else:
                    kpDict["face"]["x"] = [1.0 for point in range(0, 468)]
                    kpDict["face"]["y"] = [1.0 for point in range(0, 468)]

                keypointsDict.append(kpDict)
                #video.write(imageBGR)
                # Next frame
                ret, frame = cap.read()
            #video.release()

            height, width, channels = imageBGR.shape
            print("N° frames:",idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            glossInst = {
                    "image_dimention": {
                        "height": height,
                        "witdh": width
                    },
                    "keypoints_path": pklKeypointsPath,
                    #"image_path": pklImagePath,
                    "frame_end": idx,
                    "frame_start": 1,
                    "instance_id": IdCount,
                    "signer_id": -1,
                    "unique_name": uniqueVideoName,
                    "source": "LSP",
                    "split": "",
                    "variation_id": -1,
                    "source_video_name": videoFolderName,
                    "timestep_vide_name": videoFile[:-4]
                }

            # check if there is a gloss asigned with "word"
            glossPos = -1

            for indG, gloss in enumerate(LSP):
                if(gloss["gloss"] == word):
                    glossPos = indG

            # in the case word is in the dict
            if glossPos != -1:
                LSP[glossPos]["instances"].append(glossInst)
            else:
                glossDict = {"gloss": str(word),
                            "instances": [glossInst]
                            }
                LSP.append(glossDict)

            keypointsData = []
            imageData = []

            # 33 (pose points)
            # 21 (left hand points)
            # 21 (right hand points)
            # 87 (face mesh points)
            # * 2 (x and y axes)
            #
            # order = ‘F’ means to read / write the elements using Fortran-like index order
            # So the result of the reshape will change 
            # from => (x0,x1,x2,...,Xn, y0,y1,y2,...,Yn)
            # to => (x0,y0,x1,y1,.., Xn,Yn)
            # That means that features will have the following order: 
            # [Pose, Hand left, Hand right, face] with its corresponding size
            # keypointsData = np.asarray(list_seq).reshape(-1, (33+21+21+0)*2, order="F")

            print(videoFolderPath, videoFile, "\nkeypoints path:", pklKeypointsCompletePath)
            print("Unique name path:",os.sep.join([pklInitPath,videoFolderName,word+'_'+str(IdCount)+'.pkl']))

            # Save Pickle

            with open(pklKeypointsCompletePath, 'wb') as pickle_file:
                print()
                pkl.dump(keypointsDict, pickle_file)

            print()
            
            # Save JSON
            df = pd.DataFrame(LSP)
            df.to_json(dictPath, orient='index', indent=2)
            
            # Id of each instance
            IdCount += 1

    if not video_errors:
        print("No errors founded in any videos")
    else:
        print("\nErrors founded in:\n")
        for error in video_errors:
            print(error)


    #########################
    # CLOSE MODELS
    ##############
    holistic.close()
