# 选用Spleen作为分割对象
# 将700例ID分成350个训练集和350个测试集
# 将AVG分割的spleen的结果(2D)以及对应地原图结果(2D)进行保存
import pandas as pd
import imageio
import numpy as np
import os

ID_quanji = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/ID大全集.xlsx'
ID_quanji_list = list(pd.read_excel(ID_quanji)['ID'])
length = len(ID_quanji_list)
train_ID = ID_quanji_list[0:int(length / 2)]
print(len(train_ID))
test_ID = ID_quanji_list[int(length / 2):length]
print(len(test_ID))

import nibabel as nib
from PIL import Image

path_avg = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/ensemble/'
path = '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/segment-anything-2/inference/'
count = 0
for ID in train_ID:
    if os.path.exists(path + ID + '/mDIXON-All_BH_in_phase.nii.gz') and os.path.exists(
            path_avg + ID + '/AVG' + '/AVG_seg_' + 'pancreas' + '.nii.gz'):
        image_nii = nib.load(path + ID + '/mDIXON-All_BH_in_phase.nii.gz').get_fdata()
        AVG_nii = nib.load(path_avg + ID + '/AVG' + '/AVG_seg_' + 'pancreas.nii.gz').get_fdata()
        length_AVG = AVG_nii.shape[2]
        length_Z = image_nii.shape[2]
        if length_AVG == length_Z:
            for item in range((length_Z)):
                data_this = image_nii[:, :, item]
                label_this = AVG_nii[:, :, item]
                if len(np.unique(label_this)) > 1:
                    if not os.path.exists(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/'):
                        os.mkdir('/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/')
                    if not os.path.exists(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas'):
                        os.mkdir(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas')
                    if not os.path.exists(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/images'):
                        os.mkdir(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/images')
                    if not os.path.exists(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/masks'):
                        os.mkdir(
                            '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/masks')
                    print(count)

                    data_this = np.rot90(data_this)

                    data_this = np.uint8(data_this / np.max(data_this) * 255)


                    image = Image.fromarray(data_this)

                    # 保存为 PNG 格式
                    image.save(os.path.join(
                        '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/images/',
                        '{}.png'.format(count)), 'PNG')

                    print(count)
                    # label_this = label_this.astype(np.uint8)
                    imageio.imwrite(os.path.join(
                        '/cpfs01/projects-HDD/cfff-6117e6302119_HDD/wangchengyan/HSNet/DataSet_Segmentation/TrainDataset-pancreas/masks/',
                        '{}.png'.format(count)), label_this)
                    count = count + 1


