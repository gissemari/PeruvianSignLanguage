# data_converter.py

Use it to convert our data to TGCN and I3D data to try these models. (models are retrieved from https://dxli94.github.io/WLASL/ )

# ConvertVideoToTGCNKeypoint.py

Use it to pre-process the data before converting it to TGCN data 

# USES

1. run ConvertVideoToTGCNKeypoint.py (with its respective args) [necessarily use --hands --pose]
2. run data_converter.py (with its respective args)
3. move"data" folder into the root folder of WLASL proyect.

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

Remember to modify paths in main of train_tgcn.py  


### TGCN Library Dependency map

train_tgcn.py 
	|_.-> utils.py
	|_.-> config.py
	|_.-> tgcn_model.py
	|_.-> sign_dataset.py
	|	    |.-> utils.py
	|_.-> train_utils.py

test_tgcn.py
	|_.-> config.py
	|_.-> sign_dataset.py
	|_.-> train_utils.py
	
test.py
    |_.-> configs.py
    |_.-> sign_dataset.py
    |_.-> tgcn_model.py
    
gen_features.py

layers.py

models.py

videotransforms.py
