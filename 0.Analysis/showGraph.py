import pandas as pd
import numpy as np
import cv2

inward_ori_index = [(1, 2), (1, 3), (1, 4), (1, 5), 
                    (4, 6), (6, 8), (8, 10),
                    (5, 7), (7, 9), (9, 11),
                    (8, 12), (8, 14), (8, 16), (8, 18),
                    (9, 20), (9, 22), (9, 24), (9, 26),
                    (12, 13), (14, 15), (16, 17), (18, 19),
                    (20, 21), (22, 23), (24, 25), (26, 27)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]

data = pd.read_pickle("Data/merged/AEC-PUCP_PSL_DGI156/merge-train.pk")

for instance in data["data"]:

    for timestep in instance:

        img = np.zeros((256, 256, 3), dtype = "uint8")


        part_line = {}
        for n in range(27):
            cor_x = timestep[n][0]
            cor_y = timestep[n][1]
            if cor_x == 0 or cor_y == 0:
                continue
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 2, (0,0,255), -1)

        for start_p, end_p in inward:
            if start_p in part_line and end_p in part_line:
                start_p = part_line[start_p]
                end_p = part_line[end_p]
                cv2.line(img, start_p, end_p, (0,255,0), 2)
        
        cv2.imwrite('graph.jpg', img)
        cv2.waitKey(0)
        assert(1==2)
        #print(timestep)