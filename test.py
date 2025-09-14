import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn 

from model import direct, TTFS, rate, ann
from util import adjust_learning_rate, accuracy, AverageMeter, get_mean_and_std_1d
from noise import RandomGaussianNoise, RandomSaltPepperNoise, RandomSpeckleNoise
#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from data import MSTAR_10, MSTAR, DVSCifar10
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default='0', type=str, help='0,1,2,3')
parser.add_argument("--encode", default="p", type=str, help="Encoding [p d t ann]")
parser.add_argument("--arch", default="vgg5", type=str, help="Arch [mlp, lenet, vgg5, cifar10net]")
parser.add_argument("--dataset", default="MSTAR_10", type=str, help="Dataset [MNIST, cifar10, cifar100, MSTAR_10, dvs]")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
parser.add_argument("--resume", type=str, default = False, help="load")
parser.add_argument("--aug", default=None, type=str)

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Adversarial attack (FGSM)
def fgsm_attack(model, image, label, epsilon):

    image.requires_grad_()
    image.retain_grad()
    output = model(image)
    loss = criterion(output.to(device), label)
    model.zero_grad()
    loss.backward(retain_graph = True)
    print(image.grad)
    perturbed = epsilon * image.grad.sign()
    perturbed_image = image + perturbed
    return perturbed_image

# 데이터셋 및 모델 준비
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = MSTAR(
            root='./data/MSTAR_10',
            train = False,
            transforms=transform)

test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False, 
            num_workers=4,
            pin_memory=True)

if args.dataset == 'MNIST'or args.dataset =='MSTAR_10':
    input_dim = 1
else:
    input_dim = 3

if args.encode == 'd':
    if args.arch == 'vgg5':
        net = direct.VGG5SNN(num_cls = 10,time_step=10, in_channels = input_dim, input_size=32) # leak_mem= leak_mem
    else:
        print(f'Not implemented Err - Architecture')
        exit()

elif args.encode == 'p':
    if args.arch == 'vgg5':
        net = rate.VGG5SNN(num_cls = 10,time_step=10, in_channels = input_dim, input_size=32) # leak_mem= leak_mem
        print(f'Create new model')
    else:
        print(f'Not implemented Err - Architecture')
        exit()

elif args.encode == 't':
    if args.arch == 'vgg5':
        net = TTFS.VGG5SNN(num_cls = 10,time_step=10, in_channels = input_dim, input_size=32) # leak_mem= leak_mem
        print(f'Create new model')
    else:
        print(f'Not implemented Err - Architecture')
        exit()
        
elif args.encode == 'ann':
    if args.arch == 'vgg5':
        net = ann.VGG(vgg_name=args.arch, input_size=32, num_class=10, in_channels = input_dim)
        print(f'Create new model')
    else:
        print(f'Not implemented Err - Architecture')
        exit()
        
else:
    print(f'Not implemented Err - Encoding')
    exit()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = net.to(device)
checkpoint = torch.load(args.resume)
net.load_state_dict(checkpoint['state_dict'])
net.eval()

criterion = nn.CrossEntropyLoss().to(device)
epsilon_values = [0.01, 0.05, 0.1, 0.2]
accuracy = []

from tqdm.auto import tqdm
with tqdm(test_data_loader, unit="batch",position=0, leave=True) as pbar:
    for epsilon in epsilon_values:
        correct = 0
        total = 0
        for batch_idx, data in enumerate(pbar):    
            inputs, labels = data
            inputs = inputs.to(device)
            targets = labels.to(device)
            
            perturbed_inputs = fgsm_attack(net, inputs, targets, epsilon)
            outputs = net(perturbed_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            break  # For demonstration purposes, only test one batch
        accuracy.append(correct / total)

# 그래프 시각화
plt.figure(figsize=(8, 6))
plt.plot(epsilon_values, accuracy, marker='o')
plt.xlabel('Epsilon (Attack Strength)')
plt.ylabel('Accuracy')
plt.title('Model Robustness to FGSM Attack')
plt.grid(True)
plt.savefig(f'{args.arch}_{args.encode}_{args.aug}.png')