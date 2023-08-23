# Standard library imports
import os 

# Third party imports
import mediapipe as mp
import numpy as np
import cv2

# Local imports

def model_init(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()
    #holistic = mp_holistic.Holistic(
    #    static_image_mode= static_image_mode,
    #    min_detection_confidence=min_detection_confidence,
    #    min_tracking_confidence=min_tracking_confidence)

    return holistic

def normalization(keypoints, shoulder_distance, mid_distance):

    head_metric = shoulder_distance

    # keypoints[5] is the righ eye of the pose
    starting_point = [mid_distance[0] - 3 * head_metric, keypoints[1][5] - head_metric]
    ending_point = [mid_distance[0] + 3 * head_metric, starting_point[1] + 4.5 * head_metric]


    # Normalize individual landmarks and save the results
    for pos, kp in enumerate(keypoints[0]):
        
        # Prevent from trying to normalize incorrectly captured points
        if keypoints[0][pos] == 0.0:
            continue

        normalized_x = (keypoints[0][pos] - starting_point[0]) / (ending_point[0] - starting_point[0])
        normalized_y = (keypoints[1][pos] -   ending_point[1]) / (starting_point[1] - ending_point[1])

        keypoints[0][pos] = normalized_x
        keypoints[1][pos] = 1 - normalized_y
        
    return keypoints

def format_model_output(output):
    
    pose = output['pose']
    face = output['face']
    left_hand = output['left_hand']
    right_hand = output['right_hand']

    neck = (pose[11] + pose[12]) / 2

    newFormat = []

    newFormat.append(pose)
    newFormat.append(face)
    newFormat.append(left_hand)
    newFormat.append(right_hand)
    newFormat.append([neck])

    x = np.asarray([item[0] for sublist in newFormat for item in sublist])
    y = np.asarray([item[1] for sublist in newFormat for item in sublist])

    #body_location = [item for sublist in bType for item in sublist]
    out = np.asarray([x,y])

    # Calculate shoulder distance
    #if pose[11][0] != 0.0 or pose[12][0] != 0.0:
    #    shoulder_distance = np.linalg.norm(pose[11] - pose[12])
    #    out = normalization(out, shoulder_distance, neck)

    return out

def close_model(holistic):
    holistic.close()

def frame_process(holistic, frame):

    imageBGR = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = imageBGR.shape

    holisResults = holistic.process(imageBGR)

    kpDict = {}

    #POSE
    kpDict["pose"]={}

    if holisResults.pose_landmarks:
        kpDict["pose"] = [ [point.x, point.y] for point in holisResults.pose_landmarks.landmark]
    else:
        kpDict["pose"] = [ [0.0, 0.0] for point in range(0, 33)]
    kpDict["pose"] = np.asarray(kpDict["pose"])

    # HANDS

    
    # Left hand
    kpDict["left_hand"]={}

    if(holisResults.left_hand_landmarks):
        kpDict["left_hand"] = [ [point.x, point.y] for point in holisResults.left_hand_landmarks.landmark]
    else:
        #set the left wrist as hand points
        kpDict["left_hand"] = [ [kpDict["pose"][15][0], kpDict["pose"][15][1]] for point in range(0, 21)]
    kpDict["left_hand"] = np.asarray(kpDict["left_hand"])

    # Right hand
    kpDict["right_hand"]={}

    if(holisResults.right_hand_landmarks):
        kpDict["right_hand"] = [ [point.x, point.y] for point in holisResults.right_hand_landmarks.landmark]

    else:
        # set the rigth wrist as hand points
        kpDict["right_hand"] = [ [kpDict["pose"][16][0], kpDict["pose"][16][1]] for point in range(0, 21)]
    kpDict["right_hand"] = np.asarray(kpDict["right_hand"])

    # Face mesh
    kpDict["face"]={}

    if(holisResults.face_landmarks):

        kpDict["face"] = [ [point.x, point.y] for point in holisResults.face_landmarks.landmark]

    else:
        kpDict["face"] = [[0.0, 0.0] for point in range(0, 468)]
    kpDict["face"] = np.asarray(kpDict["face"])

    data = format_model_output(kpDict)

    return data

