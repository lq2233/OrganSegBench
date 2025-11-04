import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc


import cv2
from medpy.metric.binary import hd, hd95, dc
import pandas as pd
import imageio
import numpy as np
import os


import nibabel as nib
from PIL import Image


result_datasets = []


path_mask ='../HSNet/DataSet_Segmentation/TestDataset-saEDRV/'+'/masks/'
path_predict = './UNetoutputs/'+'EDRV'+'/'
count = 0
DICE = 0
result = []
test_ID = ['01210712V071']
for ID in test_ID:
    result_id = []
    result_id.append(ID)
    mask_files = [f for f in os.listdir('../HSNet/DataSet_Segmentation/TestDataset-saEDRV/'+'/masks/'
                                        ) if f.startswith(ID)]
    predict_files = [f for f in os.listdir('./UNetoutputs/'+ 'EDRV'+'/'
                                        ) if f.startswith(ID)]
    reference_file = path_mask+'/'+mask_files[0] # x,y
    reference_data = cv2.imread(reference_file, cv2.IMREAD_GRAYSCALE)
    Z = len(mask_files)
    x,y = reference_data.shape

    matrix_mask = np.zeros((Z, x, y))
    matrix_prediction = np.zeros((Z, x, y))


    for i in range(len(mask_files)):
        # 读出label
        mask_path = path_mask+'/'+ID+'_'+str(i)+'.png'
        # 读出pre
        prediction_path = path_predict+'/'+ID+'_'+str(i)+'.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_01 = np.where(mask < 127, 0, 1)

        pre = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        pre_01 = np.where(pre < 127, 0, 1)
        matrix_prediction[i, :, :] += pre_01
        matrix_mask[i, :, :] += mask_01


    matrix_prediction[matrix_prediction > 0] = 1
    matrix_mask[matrix_mask > 0] = 1


    dice = dc(matrix_prediction, matrix_mask)
    result_id.append(dice)
    result.append(result_id)
    print(ID, dice)
    DICE = DICE + dice
AVG_DSC = DICE/len(test_ID)
result_datasets.append(AVG_DSC)
print('AVG DSC:', AVG_DSC)