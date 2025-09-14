import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
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

class MSTAR_10(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes = self.read_data_set()
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.image_files_path[index])
        image = image.convert("L")

        if self.transforms is not None:
            image = self.transforms(image)

        return {'image': image, 'label': self.labels[index]}

    def __len__(self):
        return self.length
    
import random
class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.rotate = transforms.RandomRotation(degrees=30)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-30, 30))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        data = self.resize(data.permute([3, 0, 1, 2]))

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))

def build_dvscifar(path='data/cifar-dvs', transform=False):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=False)
    val_dataset = DVSCifar10(root=val_path, transform=False)

    return train_dataset, val_dataset

class KITTI(Dataset):
    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names)

    def __init__(self):

        self.image_path = '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/dataset/flyingthing3d_0920/train/flying_img/'
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find("png") != -1 if not x.startswith('.')]
        # minjung
        # self.dataset_path = '/media/ellaroine/58789D3B5C406636/datasets/FlyingThings3D/GANet_disparity/flying_ganet_all/' 
        # jini
        self.dataset_path = '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/dataset/flyingthing3d_0920/train/flying_psm/' 
        # self.dataset_path = '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/dataset/flyingthing3d_0920/train/flying_ganet/' 
        dataset_names = sorted(listdir(self.dataset_path))
        disp_names = [x for x in dataset_names if x.find("png") != -1 if not x.startswith('.')]
        # minjung
        # self.gt_disp_path = '/media/ellaroine/58789D3B5C406636/datasets/FlyingThings3D/flyingthings3d__disparity/disparity/flying_disp_all/' #'/media/ellaroine/58789D3B5C406636/datasets/cityscape/gt_reshape/aachen'
        # jini
        self.gt_disp_path = '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/dataset/flyingthing3d_0920/train/flying_disp/' #'/media/ellaroine/58789D3B5C406636/datasets/cityscape/gt_reshape/aachen'
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find("png") != -1 if not x.startswith('.')] 
               
        self.image_names = image_names[:26080]
        self.disp_names = disp_names[:26080]
        self.gt_disp_names = gt_disp_names[:26080]

        
    def __len__(self):
        return len(self.image_names)

    # def img_save(self,x,i):
    #     save_image(x, '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/PiT_Adapter_loss/checkpoints/global/input_rgb/input_rgb_%07d.png' %(i))
    #     i += 1

    #     return i

    # def disp_im_save(self,x,i):
    #     save_image(x, '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/PiT_Adapter_loss/checkpoints/global/input_disp/input_disp_%07d.png' %(i))
    #     # f = open('/media/ellaroine/58789D3B5C406636/paper1/output_img/input_disp/input_disp_%07d.txt' %(i), 'a')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # f.close()
    #     i += 1

    #     return i

    # def gt_disp_im_save(self,x,i):
    #     save_image(x, '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/PiT_Adapter_loss/checkpoints/global/gt_disp/gt_disp_%07d.png' %(i))
    #     # f = open('/media/ellaroine/58789D3B5C406636/paper1/output_img/gt_disp/gt_disp_%07d.txt' %(i), 'a')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # f.close()
    #     i += 1

    #     return i

    # def gt_conf_im_save(self,x,i):
    #     save_image(x, '/media/cvlab/0eadf366-1eab-42b3-abd5-5932713d6b6e/PiT_Adapter_loss/checkpoints/global/gt_conf/gt_conf_%07d.png' %(i))
    #     # f = open('/media/ellaroine/58789D3B5C406636/paper1/output_img/gt_conf/gt_conf_%07d.txt' %(i), 'a')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # np.savetxt(f, img1[0].detach().cpu().numpy(), fmt='%.4f')
    #     # f.close()
    #     i += 1

    #     return i
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        _, hei, wei = imag.size() 
        
        disp_name = os.path.join(self.dataset_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
        disp = disp/240.
        # print(disp.shape) -> 1, 540,960
        
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)
        # gt_disp = gt_disp/256.
        
        gt_conf = (torch.abs(disp-gt_disp) <= 7/240. ).type(dtype=torch.float) # 먼저 값 범위 확인 확실히 넣기
        gt_conf[gt_disp==0] = -1
                        
        new_hei = 224 #240
        new_wei = 224 #776
        
        top = np.random.randint(0,hei-new_hei)
        left = np.random.randint(0,wei-new_wei)
        disp = disp[:,top:top+new_hei,left:left+new_wei]
        imag = imag[:,top:top+new_hei,left:left+new_wei]
        gt_disp = gt_disp[:,top:top+new_hei,left:left+new_wei]
        gt_conf = gt_conf[:,top:top+new_hei,left:left+new_wei]
        
       
        # tmp_disp = disp.squeeze(0).detach().cpu().numpy()[:,:,0]
        # tmp_gt_disp = gt_disp.squeeze(0).detach().cpu().numpy()[:,:,0]
        # # tmp_gt_conf= gt_conf.squeeze(0).detach().cpu().numpy()
        # # # # f2 = np.vectorize(tmp_disp)
        # # # # fig = plt.plot(tmp_disp)
        # # # # fig = plt.imread(tmp_disp)
        
        # plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
        # plt.imshow(tmp_disp)
        # plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
        # plt.imshow(tmp_gt_disp)
        # # plt.subplot(3, 1, 3)
        # # plt.imshow(tmp_gt_conf)
        # plt.show()
        
        # # def f(x):
        # #     return int(x)
        # # f2 = np.vectorize(f)
        # # x = tmp_disp
        # # plt.plot(f2(x))
        # # plt.show()
        
        # # print(disp.size())
        # # print(disp[0,112,112])
        # # print(torch.max(disp))
        # # print(torch.min(disp))
        # # print("--------------------")
        # # print(gt_disp.size())
        # # print(gt_disp[0,112,112])
        # # print(torch.max(gt_disp))
        # # print(torch.min(gt_disp))
        # # print("--------------------")
        # # print(gt_conf.size())
        # # print(gt_conf[0,112,112])
        # # print(torch.max(gt_conf))
        # # print(torch.min(gt_conf))
        # print("--------------------")
        # print(tmp_disp.shape)
        # print(disp.shape)
        # print(imag.shape)
        # print(gt_conf.shape)
        

        # print("0")
        return {'disp': disp, 'imag': imag, 'gt_conf': gt_conf, 'gt_disp': gt_disp}

    