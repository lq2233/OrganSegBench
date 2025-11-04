import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import HSNet
from utils.dataloader import test_dataset
import cv2
from medpy.metric.binary import dc, hd, hd95
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth_pancreas_for_test/PolypPVT_pancreas/9PolypPVT-pancreas-best.pth') # 模型路径,需要修改
    opt = parser.parse_args()
    model = HSNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    print('begin')
    # for _data_name in ['pancreas-ID']:
    ##### put data_path here #####
    data_path = './DataSet_Segmentation/TestDataset-pancreas/'
    ##### save_path #####
    save_path = './result_map/PolypPVT-pancreas/'
    if not os.path.exists('./result_map/'):
        os.makedirs('./result_map/')
    if not os.path.exists('./result_map/PolypPVT-pancreas/'):
        os.makedirs('./result_map/PolypPVT-pancreas/')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    for i in range(num1):
        print(i)
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        P1, P2, P3, P4 = model(image)
        res = F.upsample((P1+P2+P3+P4)/4, size=gt.shape, mode='bilinear', align_corners=False)  # Avg fusion of the four outputs
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res[res>=0.5] = 1
        res[res<0.5] = 0
        gt[gt>=0.5] = 1
        gt[gt<0.5] = 0
        print(dc(res, gt))
        cv2.imwrite(save_path+name, res*255)
    print('Finish!')
