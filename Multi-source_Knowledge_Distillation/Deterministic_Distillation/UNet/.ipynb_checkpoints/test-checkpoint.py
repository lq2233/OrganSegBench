import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from unet import Unet
from dataloader import test_dataset
import cv2
from medpy.metric.binary import dc, hd, hd95
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_path_EDRV_for_test/EDRV/EDRV_model_dict_best_18_0.811.pth') # 模型路径,需要修改
    opt = parser.parse_args()
    net = Unet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], initializers={'w':'he_normal', 'b':'normal'}, apply_last_layer=True, padding=True)
    net.load_state_dict(torch.load(opt.pth_path))
    net.cuda()
    net.eval()
    print('begin')
    
    # for _data_name in ['EDRV-AVG-ID']:
    ##### put data_path here #####
    
    data_path = '../HSNet/DataSet_Segmentation/TestDataset-saEDRV/' # Same masks (AVG fusion masks) for HSNet and UNet training
    ##### save_path #####
    save_path = './UNetoutputs/EDRV/'
    if not os.path.exists('./UNetoutputs/'):
        os.makedirs('./UNetoutputs/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    for i in range(num1):
        print(i)
        image, gt, name, size = test_loader.load_data()
        print(size)
        image = image.to('cuda')
        
        # image.cuda()
        
        pred = net.forward(image, False)
                
        pred = F.interpolate(
                    pred, 
                    size=size[::-1],  # (height, width)
                    mode='bilinear', 
                    align_corners=False
                )
        print(pred.shape)

                
        pred = pred.sigmoid().data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0
        gt = np.array(gt)
        gt[gt>=0.5] = 1
        gt[gt<0.5] = 0
        
        print(save_path+name, dc(pred, gt))
        cv2.imwrite(save_path+name, pred*255)
                
    print('Finish!')
