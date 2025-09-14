import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import os
from os import listdir
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image


from torchvision.utils import save_image

class MSTAR(Dataset):
    def __init__(self, root, train, transforms=None):
        self.transforms = transforms

        self.train = train
        
        if self.train:
            self.data = datasets.ImageFolder(root+"/train")
        else:
            self.data = datasets.ImageFolder(root+"/test")
        #print(self.data.imgs)
        self.img, self.targets = zip(*self.data.imgs)
        #print(self.img, self.targets)
        #print(self.data[0])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #img = io.imread(self.data[index], as_gray=True)
        #img = Image.fromarray(img)
        image = Image.open(self.img[index])
        image = image.convert("L")
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        target = self.targets[index]
        
        return image, target

class MSTAR_10(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]
        #print(class_names)
        for index, class_name in enumerate(class_names):
            label = index
            #print(label)
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                #print(img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
       self.data_set_path = data_set_path
       self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
       #print(self.image_files_path, self.labels, self.length, self.num_classes)
       self.transforms = transforms
    
    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("L")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

        #return (image, self.labels[index])

    def __len__(self):
        return self.length


import random
class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.train = train

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        data = self.resize(data.permute([0, 3, 1, 2]))
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))