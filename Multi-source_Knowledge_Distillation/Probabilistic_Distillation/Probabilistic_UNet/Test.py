# loads a trained model and saves some results on the disk

# The trained model dict is loaded from directory 'cpk_directory' and results are saved in 'out_dir/visual_results'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from load_organ_data import OrganSegData_test_New
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pickle
import os
import re
import cv2
from medpy.metric.binary import dc, hd, hd95
# checkpoint directory
cpk_directory = './Model_path_test/AO/'     # checkpoint saved here by Train_Model.py
print('Using the trained model from directory: ', cpk_directory)
if not os.path.exists(cpk_directory):
    raise ValueError('Please specify the out_dir in visualize.py which contains the trained model dict')

out_dir = './Output/AO/'  # results will be saved in 'out_dir/visual_results'


batch_size_val = 1 
save_batches_n = 10e5     # save this many batches
samples_per_example = 10

dataset = OrganSegData_test_New(dataset_location = './Data_Sample/TestDataset/AO/')
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(1 * dataset_size))

train_indices, test_indices = indices[split:], indices[:split]
test_sampler = SubsetRandomSampler(test_indices)
# test_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=test_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size_val)
print("Number of test patches:", len(test_indices))

# network
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=6, no_convs_fcomb=3, beta=10.0)
net.cuda()

# load pretrained model
cpk_name = os.path.join(cpk_directory, 'AO_model_dict_999_final.pth')
net.load_state_dict(torch.load(cpk_name))

net.eval()
with torch.no_grad():
    result_DICE_all = []
    for step, (patch, mask, image_path, label_dir) in enumerate(test_loader):
        # print(image_path)
        pred_result = []
        label_result = []
        if step >= save_batches_n:
            break
        patch = patch.cuda()
        mask = mask.cuda()
        mask = torch.unsqueeze(mask,1)
        output_samples = []
        for i in range(samples_per_example): 
            net.forward(patch, None, training=False)
            output_samples.append(torch.sigmoid(net.sample(testing=True)).detach().cpu().numpy())

        for k in range(patch.shape[0]):    # for all items in batch
            image_path1 = image_path[k]

            patch_out = patch[k, 0, :,:].detach().cpu().numpy()
            mask_out = mask[k, 0, :,:].detach().cpu().numpy()

            
            
            result = []
            pred_result_temp = []
            label_result_temp = []
            for j in range(len(output_samples)):  # for all output samples
                pred = output_samples[j][k, 0, :, :]
                
                pred_result_temp.append(pred)

            prob = torch.mean(torch.stack([torch.as_tensor(x) for x in pred_result_temp]), dim=0).numpy()
            pred = (prob >= 0.5).astype(np.uint8)
            
            if 'A-0' in image_path1:
                ID_get = re.search(r'(\d{8}V\d{3})_(\d+)', image_path1).group(1)
                ID_get = 'A-' + ID_get
            else:
                ID_get = re.search(r'(\d{8}V\d{3})_(\d+)', image_path1).group(1)
            item_get = re.search(r'(\d{8}V\d{3})_(\d+)', image_path1).group(2)

            if not os.path.exists('./Output/AO/'):
                os.mkdir('./Output/AO/')
            if not os.path.exists('./Output/AO/'+ID_get):
                os.mkdir('./Output/AO/'+ID_get)
            
            cv2.imwrite('./Output/AO/'+ID_get+'/'+ID_get+'_'+item_get+'.png', pred * 255)
