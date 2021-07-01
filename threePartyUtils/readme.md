# data_converter.py

Use this py to convert our data to TGCN and I3D data (retrieved from https://dxli94.github.io/WLASL/ ) to try these models.


# USES

1. move "data_converter.py" and "data" folder into the root folder of WLASL proyect.
2. run data_converter.py (with its respective args)

# Data structure

## pose_per_individual_videos

'''
{'version': 1.3 ,
                       'people':[{'person_id':[0],
                                  'pose_keypoints_2d':[ 75 variables ],
                                  'hand_left_keypoints_2d':[63 variables],
                                  'face_keypoints_2d':[],
                                  'hand_right_keypoints_2d':[63 variables],
                                  'face_keypoints_3d':[],
                                  'hand_left_keypoints_3d':[],
                                  'hand_right_keypoints_3d':[]
                                }]
                       }
'''

### keypoints structure

'###_keypoints_2d':[ x0, y0, p0, x1, y1, p1, x2, y2, p2, ... xn, yn, pn]

where necessary
  
	- x is "x" axis (without normalization)
      	- y is "y" axis (without normalization)
      	- p is probability 

## splits

'''
{
        "gloss": "ESE",
        "instances": [
            {
                "bbox": [
                    137,
                    16,
                    492,
                    480
                ],
                "frame_end": 8,
                "frame_start": 1,
                "instance_id": 0,
                "signer_id": 0,
                "source": "LSP",
                "split": "train",
                "url": "",
                "variation_id": 0,
                "video_id": "ESE_555"
            },
           ...]
            
'''

### WLASL_VID

videos extracted from our proyect

# P.D.

Remember to modify patha in main of train_tgcn.py  

