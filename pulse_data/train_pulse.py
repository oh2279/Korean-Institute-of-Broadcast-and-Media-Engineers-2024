import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WORLD_SIZE"] = "1"
import shutil
import logging
import datetime

from model import direct_vggsnn_1d, direct_vggann_1d
from util import adjust_learning_rate, accuracy, AverageMeter, get_mean_and_std_1d

from pulse_dataset import PulseDataset

from tqdm.auto import tqdm

import numpy as np

import sys
import time
import argparse

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision
from torchvision import transforms

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', default='0', type=str, help='0,1,2,3')
parser.add_argument('--dump-dir', type=str, default="log")
parser.add_argument("--encode", default="d", type=str, help="Encoding [p d t ann]")
parser.add_argument("--arch", default="vgg9", type=str, help="Arch [mlp, lenet, vgg5, cifar10net, vgg9, vgg11]")
parser.add_argument("--dataset", default="pulse", type=str, help="Dataset [pulse]")
parser.add_argument("--optim", default='adamw', type=str, help="Optimizer [adamw, sgd]")
parser.add_argument('--leak_mem',default=0.5, type=float)
parser.add_argument('--T', type=int, default=4)
parser.add_argument('--epoch', type=int, default=50)
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

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)
    
logger.warning(
    f"device: {device}, "
    f"gpu: {args.gpu}, "
    )

logger.info(dict(args._get_kwargs()))

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar', epoch=1):
    filepath = os.path.join(checkpoint, str(epoch)+'_'+filename)
    if is_best:
        torch.save(state, filepath)
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def train(epoch, net, train_data_loader, optimizer, criterion_cls, criterion_reg, device):
    net.train()
    train_loss_list = AverageMeter()
    train_acc_list = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(train_data_loader, unit="batch",position=0, leave=True) as tepoch:
        for batch_idx, data in enumerate(tepoch):      
            inputs, labels1, labels2 = data
            inputs = inputs.to(device)
            targets_cls = labels1.to(device)
            targets_reg = labels2.to(device)
            
            optimizer.zero_grad()

            output1, output2, output3, output4, output5, output6 = net(inputs)
            loss1 = criterion_cls(output1, targets_cls[:, 0])
            loss2 = criterion_cls(output2, targets_cls[:, 1])
            loss3 = criterion_reg(output3, targets_reg[:, 0].unsqueeze(1))
            loss4 = criterion_reg(output4, targets_reg[:, 1].unsqueeze(1))
            loss5 = criterion_reg(output5, targets_reg[:, 2].unsqueeze(1))
            loss6 = criterion_reg(output6, targets_reg[:, 3].unsqueeze(1))
            
            # Classification과 Regression Loss를 각각 합친 뒤 더함
            loss_cls = loss1 + loss2
            loss_reg = loss3 + loss4 + loss5 + loss6

            # loss_reg가 너무 클 수 있으니 weight를 더 작게 설정
            loss = loss_cls + 0.001 * loss_reg
            loss.backward()
            optimizer.step()
            
            train_loss_list.update(loss.item(), inputs.size(0))
            
            #prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            #top1.update(prec1.item(), inputs.shape[0])
            #top5.update(prec5.item(), inputs.shape[0])

            #tepoch.set_description("Train Iter: {batch}/{iter}. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
            tepoch.set_description("Train Iter: {batch}/{iter}. Loss: {loss:.4f}. ".format(
                batch=batch_idx + 1,
                iter=len(train_data_loader),
                loss=train_loss_list.avg,
                #top1=top1.avg,
                #top5=top5.avg,
            ))
                
    tepoch.close()
    
    #logger.info("top-1 acc: {:.2f}".format(top1.avg))
    #logger.info("top-5 acc: {:.2f}".format(top5.avg))
    
    return train_loss_list.avg, top1.avg

def test(epoch, net, test_data_loader, criterion_cls, criterion_reg, device):
    net.eval()
    test_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with tqdm(test_data_loader, unit="batch",position=0, leave=True) as pbar:
        with torch.no_grad():
            for batch_idx, data in enumerate(pbar):         
                inputs, labels1, labels2 = data
                inputs = inputs.to(device)
                targets_cls = labels1.to(device)
                targets_reg = labels2.to(device)

                # forward
                out1, out2, out3, out4, out5, out6 = net(inputs)

                # loss 계산
                loss1 = criterion_cls(out1, targets_cls[:, 0])
                loss2 = criterion_cls(out2, targets_cls[:, 1])
                loss3 = criterion_reg(out3, targets_reg[:, 0].unsqueeze(1))
                loss4 = criterion_reg(out4, targets_reg[:, 1].unsqueeze(1))
                loss5 = criterion_reg(out5, targets_reg[:, 2].unsqueeze(1))
                loss6 = criterion_reg(out6, targets_reg[:, 3].unsqueeze(1))

                # 전체 loss
                # Classification과 Regression Loss를 각각 합친 뒤 더함
                loss_cls = loss1 + loss2
                loss_reg = loss3 + loss4 + loss5 + loss6

                # loss_reg가 너무 클 수 있으니 weight를 더 작게 설정
                loss = loss_cls + 0.001 * loss_reg

                test_losses.update(loss.item(), inputs.shape[0])
                #prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                #top1.update(prec1.item(), inputs.shape[0])
                #top5.update(prec5.item(), inputs.shape[0])

                #pbar.set_description("Test Iter: {batch}/{iter}. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                pbar.set_description("Test Iter: {batch}/{iter}. Loss: {loss:.4f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_data_loader),
                    loss=test_losses.avg,
                    #top1=top1.avg,
                    #top5=top5.avg,
                ))

    return test_losses.avg, top1.avg

def main():
    batch_size = args.batch_size
    lr = args.lr

    dataset_dir = '/home/gpuadmin/papers/beomseok/Broadcast_and_Media/augmentation/pulse_data/pulse_raw'
    dump_dir = args.dump_dir

    arch_prefix = args.dataset +"_" + args.arch + "_" + args.encode + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    if args.optim == 'sgd': args.lr = 0.01
    elif args.optim == 'adamw': args.lr = 0.001
    
    if args.encode == 'ann':
        file_prefix = "batch_size" + str(args.batch_size) + "_lr" + str(args.lr) + "_epoch" + str(args.epoch) + "_" + str(args.aug) +  "_" + str(args.optim)
    else:
        file_prefix = "T" + str(args.T) + "_batch_size" + str(args.batch_size)  + "_lr" + str(args.lr) + "_epoch" + str(args.epoch) + "_" + str(args.aug) +  "_" + str(args.optim)

    print('{}'.format(args.setting))

    print("arch : {} ".format(arch_prefix))
    print("hyperparam : {} ".format(file_prefix))

    log_dir = os.path.join(dump_dir, 'logs', arch_prefix, file_prefix)
    model_dir = os.path.join(dump_dir, 'models', arch_prefix, file_prefix)
    log_list_dir = os.path.join(dump_dir, 'log_list', arch_prefix, file_prefix)
    file_prefix = file_prefix + '.pkg'

    if args.log == 'True':
        print('Make log directory')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            args.writer = SummaryWriter(log_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_list_dir):
            os.makedirs(log_list_dir)

    T = args.T
    N = args.epoch

    file_prefix = 'lr-' + np.format_float_scientific(lr, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'

    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO)

    # Data augmentation
    mean = {}
    std = {}


    # 임시설정    
    if args.dataset == 'pulse':
        input_dim = 5
        num_cls = 2

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        label_excel_path = '/home/gpuadmin/papers/beomseok/Broadcast_and_Media/augmentation/pulse_data/label.xlsx'
        dataset = PulseDataset(
            data_dir=dataset_dir,
            label_excel_path=label_excel_path,
            channel=input_dim,
            flat_interval=(7000, 43000),
            transform=transform)


    # dataset 분할
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
      
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
        
    if args.encode == 'd':
        net = direct_vggsnn_1d.VGG9SNN(num_cls = num_cls,time_step=T, in_channels = input_dim, input_size=36000) 
    elif args.encode == 'ann':
        net = direct_vggann_1d.VGG9ANN(vgg_name = args.arch, input_size=36000, num_class=num_cls, in_channels = input_dim)
    else:
        print(f'Not implemented Err - Encoding')
        exit()
    
    #print(device)
    net= net.to(device)
    
    # Configure the loss function and optimizer
    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_reg = nn.MSELoss().to(device)
    
    if args.optim == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=0.001,betas=(0.9, 0.999), weight_decay=1e-2)
        
    best_acc = 0
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    
    #lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,first_cycle_steps=100,cycle_mult=1.0,max_lr=0.01,min_lr=0.0001,warmup_steps=50,gamma=0.5)
            
    # Training Loop
    summary(net, (input_dim, 36000))
    print(net)

        
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}")
    logger.info(f"  Num Epochs = {args.epoch}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Simulation # time-step = {T}")
    logger.info(f"  Learning rate     = {args.lr}")

    best_acc = 0
    start_epoch = 0
    net.zero_grad()

    print("train_data_loader", len(train_data_loader))
    print("test_data_loader", len(test_data_loader))
    #return 0
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy= []
    for epoch in range(start_epoch, args.epoch):
        train_loss, train_acc = train(epoch, net, train_data_loader, optimizer, criterion_cls, criterion_reg, device)
        test_loss, test_acc = test(epoch, net, test_data_loader, criterion_cls, criterion_reg, device)
        
        lr_scheduler.step()
        
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        if args.log:
            args.writer.add_scalar('train/1.train_loss', train_loss, epoch)
            args.writer.add_scalar('train/2.train_acc', train_acc, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, model_dir, epoch=epoch + 1)

        print("Epoch: {}, Train Loss: {}, Train Acc: {}, Test Loss: {}, Test Acc: {}, Best Acc: {}".format(
            epoch, train_loss, train_acc, test_loss, test_acc, best_acc
        ))
        
        print('lr : ', lr_scheduler.get_last_lr())
    # best_acc 를 txt파일로 저장
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)
        
        with open(os.path.join(log_list_dir, 'logs.txt'), 'w') as f:
            for a,b,c,d in zip(train_losses, test_losses, train_accuracy, test_accuracy):
                f.write(f'{a:.4f}, {b:.4f}, {c:.4f}, {d:.4f}\n')
    
if __name__=="__main__":
    main()
