import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WORLD_SIZE"] = "1"
import shutil
import logging
import datetime

from model import direct, TTFS, rate, ann
from util import adjust_learning_rate, accuracy, AverageMeter, get_mean_and_std_1d
from noise import RandomGaussianNoise, RandomSaltPepperNoise, RandomSpeckleNoise
#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from data import MSTAR_10, MSTAR, DVSCifar10

from tqdm.auto import tqdm

import numpy as np

import sys
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', default='0', type=str, help='0,1,2,3')
parser.add_argument('--dump-dir', type=str, default="newlog")
parser.add_argument("--encode", default="d", type=str, help="Encoding [p d t ann]")
parser.add_argument("--arch", default="vgg9", type=str, help="Arch [mlp, lenet, vgg9, cifar10net]")
parser.add_argument("--dataset", default="MSTAR_10", type=str, help="Dataset [MNIST, cifar10, cifar100, MSTAR_10, dvs]")
parser.add_argument("--optim", default='sgd', type=str, help="Optimizer [adamw, sgd]")
parser.add_argument('--leak_mem',default=0.5, type=float)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--train_display_freq", default=1, type=int, help="display_freq for train")
parser.add_argument("--test_display_freq", default=1, type=int, help="display_freq for test")
parser.add_argument("--setting", type=str, help="display_freq for test")
parser.add_argument("--resume", type=str, default = False, help="load")
parser.add_argument("--log", default='True', type=str, help="[True, False]")
parser.add_argument("--aug", type=str, default = 'None', help="augmentation methods")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision
from torchvision import transforms

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    # Data augmentation
    img_size = {
        'MNIST' : 28,
        'MSTAR_10': 32,
        'cifar10': 32,
        'cifar100': 32,
        'dvs': 48
    }

    num_cls = {
        'MNIST' : 10,
        'MSTAR_10': 10,
        'cifar10': 10,
        'cifar100': 100,
        'dvs': 10,
    }

    mean = {
        'MNIST' : 0.1307,
        'MSTAR_10': 0.1413,
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        }

    std = {
        'MNIST' : 0.3081,
        'MSTAR_10': 0.1126,
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        }

    if args.dataset == 'MNIST'or args.dataset =='MSTAR_10':
        input_dim = 1
    elif args.dataset == 'dvs':
        input_dim = 2
    else:
        input_dim = 3

    dataset_dir = './data/' + args.dataset
    batch_size = 1
    
    if args.dataset == 'MSTAR_10':
        #color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        transform_train = transforms.Compose([
            transforms.Resize((32,32)),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            #transforms.AugMix(), 
            #transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            #transforms.RandAugment(),
            #RandomGaussianNoise(),
            #RandomSaltPepperNoise(),
            #RandomSpeckleNoise(),
            #transforms.Normalize(0.5,0.5),
            #transforms.RandomErasing(),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5),
        ])
        
        train_dataset = MSTAR(
            root=dataset_dir,
            train = True,
            transforms=transform_train)
        test_dataset = MSTAR(
            root=dataset_dir,
            train = False,
            transforms=transform_test)
        
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True, 
            num_workers=4,
            pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False, 
            num_workers=4,
            pin_memory=True)

    elif args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            #transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            #transforms.AugMix(), 
            #transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            #transforms.RandAugment(),
            #RandomGaussianNoise(p=0.2),
            #RandomSaltPepperNoise(),
            #RandomSpeckleNoise(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            #transforms.Normalize(mean[args.dataset], std[args.dataset]),
            #transforms.RandomErasing(),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean[args.dataset], std[args.dataset])
        ])
        
        #Sampling()

        train_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            transform=transform_train,
            download=False)
            
        test_dataset = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            transform=transform_test,
            download=False)
    

    import random
    # 데이터셋에서 무작위로 하나의 이미지 선택
    image, _ = train_dataset[1]

    # 텐서를 PIL 이미지로 변환
    image = transforms.ToPILImage()(image)

    # 이미지 저장
    image.save(f'{args.dataset}_{args.aug}.png')
   
if __name__=="__main__":
    main()
    #wandb.agent(sweep_id, function=main, count=10)