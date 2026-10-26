import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from PIL import Image
import pickle
import torchvision.transforms as transforms
from torchvision.utils import save_image
#import dicom
  
    

class OrganSegData_test_New(Dataset):

    def __init__(self, dataset_location, transform=None):
        self.dataset_location = dataset_location
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        

        for root, dirs, files in os.walk(dataset_location):

            image_files = [f for f in files if (f.endswith('.png') or f.endswith('.jpg')) and 'image' in f]
            label_files = [f for f in files if f.endswith('.png') and 'label' in f]
            image_files.sort()  #
            label_files.sort()
            # print(label_files)
            if image_files:
                label_paths = []
                # 假设每个子目录中有多个图像和标签文件，存储路径
                for image_file in image_files:
                    image_path = os.path.join(root, image_file)
                    self.image_paths.append(image_path)

                for label_file in label_files:
                    label_path = os.path.join(root, label_file)
                    label_paths.append(label_path)
                    self.label_paths.append(sorted(label_paths))

        assert len(self.image_paths) == len(self.label_paths), "每个图像必须有对应的标签"
    
    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label_dir = self.label_paths[index][0]

        image = Image.open(image_path).convert('L')  # 转为灰度图
        image = np.array(image)


        label = Image.open(label_dir).convert('L')  # 转为灰度图
        label = np.array(label)
        

        image = torch.from_numpy(image)

        label = torch.from_numpy(label)
        image = image.type(torch.FloatTensor).unsqueeze(0)
        label = label.type(torch.FloatTensor)
        image /= 255.0
        label /= 255.0


        if self.transform is not None:
            image = self.transform(image)


        return image, label, image_path, label_dir

    def __len__(self):

        return len(self.image_paths)
    
    
class OrganSegData_train_New(Dataset):
    

    def __init__(self, dataset_location, transform=None):
        
        self.dataset_location = dataset_location
        self.transform = transform
        self.image_paths = []
        self.label_paths = []
        

        for root, dirs, files in os.walk(dataset_location):

            image_files = [f for f in files if (f.endswith('.png') or f.endswith('.jpg')) and 'image' in f]
            label_files = [f for f in files if f.endswith('.png') and 'label' in f]
            image_files.sort()
            label_files.sort()
            
            # print(label_files)
            
            if image_files:
                label_paths = []

                for image_file in image_files:
                    image_path = os.path.join(root, image_file)
                    self.image_paths.append(image_path)

                for label_file in label_files:
                    label_path = os.path.join(root, label_file)
                    label_paths.append(label_path)
                self.label_paths.append(sorted(label_paths))

        assert len(self.image_paths) == len(self.label_paths), "每个图像必须有对应的标签"
    
    def __getitem__(self, index):

        image_path = self.image_paths[index]
        label_dir = self.label_paths[index][random.randint(0,5)]  # Randomly Select the masks generalized by 6 SFMs


        image = Image.open(image_path).convert('L')
        image = np.array(image)


        label = Image.open(label_dir).convert('L')  #
        label = np.array(label)
        

        image = torch.from_numpy(image)

        label = torch.from_numpy(label)
        image = image.type(torch.FloatTensor).unsqueeze(0)
        label = label.type(torch.FloatTensor)
        image /= 255.0
        label /= 255.0


        if self.transform is not None:
            image = self.transform(image)


        return image, label, image_path, label_dir

    def __len__(self):
        # 返回数据集的大小
        return len(self.image_paths)
