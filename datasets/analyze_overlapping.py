import os
import pandas as pd
import sys
import numpy as np

labels_hard = ['birds_singing', 'car', 'people talking', 'footsteps', 
               'children voices', 'wind_blowing', 'brakes_squeaking', 
               'large_vehicle', 'cutlery and dishes', 'metro approaching', 
               'metro leaving']

for event in labels_hard:
    print(event)
    event_overlap = np.zeros((1,5))
    meta_folder_1 = '/mnt/nfs2/hanyin/LASS4SED/datasets/maestro_real/development_annotation'
    for place in os.listdir(meta_folder_1):
        meta_folder_2 = meta_folder_1 + f'/{place}'
        for metaname in os.listdir(meta_folder_2):
            meta_path = meta_folder_2 + f'/{metaname}'

            meta_data = pd.read_csv(meta_path, delimiter='\t', header=None)
            # print(meta_data.shape) # [N, 4]
            meta_data_1 = meta_data[meta_data.iloc[:,3] >= 0.5] # all events
            # print(meta_data_1)
            meta_data_2 = meta_data_1[meta_data_1.iloc[:,2] == event] # located event
            meta_data_3 = meta_data_1[meta_data_1.iloc[:,2] != event] # located event
            # print(meta_data_2)
            
            nums = meta_data_2.shape[0]
            for i in range(nums):
                onset = meta_data_2.iloc[i, 0]
                offset = meta_data_2.iloc[i, 1]

                meta_data_4 = meta_data_3[meta_data_3.iloc[:,0] == onset]
                meta_data_5 = meta_data_4[meta_data_4.iloc[:,1] == offset]
                # print(meta_data_3.shape)
                overlap_num = meta_data_5.shape[0]
                event_overlap[0, overlap_num] += 1

            
            # sys.exit()
    print(np.sum(event_overlap))
    print(event_overlap/np.sum(event_overlap)*100)