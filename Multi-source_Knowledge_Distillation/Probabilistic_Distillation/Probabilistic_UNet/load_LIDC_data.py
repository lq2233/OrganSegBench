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

class LIDC_IDRI(Dataset):
    images = []
    labels = []
    # series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            # print(filename)
            if '.pickle' in filename and 'spleen' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            # print(key)
            # print('******************************************************************************************')
            # print(np.min(np.unique(value['image'])), np.max(np.unique(value['image'])))
            # print(type(value['image']))
            # print(type(value['masks']), len(value['masks']), type(value['masks'][0]), value['masks'][0].shape)
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            # self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        
        # label = self.labels[index][random.randint(0,3)].astype(float)
        label = self.labels[index][0].astype(float)
        
        if self.transform is not None:
            image = self.transform(image)

        # series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        # return image, label, series_uid
        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
    
    
class LIDC_IDRI_test(Dataset):
    images = []
    labels = []
    # series_uid = []

    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        max_bytes = 2**31 - 1
        data = {}
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            # print(filename)
            if '.pickle' in filename and 'spleen' in filename and 'test' in filename:
                print("Loading file", filename)
                file_path = dataset_location + filename
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)
        
        for key, value in data.items():
            # print(key)
            # print('******************************************************************************************')
            # print(np.min(np.unique(value['image'])), np.max(np.unique(value['image'])))
            # print(type(value['image']))
            # print(type(value['masks']), len(value['masks']), type(value['masks'][0]), value['masks'][0].shape)
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            # self.series_uid.append(value['series_uid'])

        assert (len(self.images) == len(self.labels))

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)

        #Randomly select one of the four labels for this image
        
        # label = self.labels[index][random.randint(0,3)].astype(float)
        label = self.labels[index][0].astype(float)
        
        if self.transform is not None:
            image = self.transform(image)

        # series_uid = self.series_uid[index]

        # Convert image and label to torch tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        #Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)

        # return image, label, series_uid
        # print(image.shape, label.shape)
        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)
    
    
    
    
class LIDC_IDRI_test_New(Dataset):

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
    
    
class LIDC_IDRI_train_New(Dataset):
    

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
