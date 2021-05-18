#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import TrainValDataset
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *

parser = argparse.ArgumentParser(description="model_train")
parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[120000,180000], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="final_H", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=10000,help='save intermediate model')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
parser.add_argument("--iter_val", type=int, default=1000, help='iter val')
parser.add_argument("--iter_epoch", type=int, default=500, help='iter epoch')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
if not os.path.isdir(opt.save_path):
    os.makedirs(opt.save_path)

def main():
    print('Loading dataset ...\n')

    dataset_train = TrainValDataset("train")
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    dataset_val = TrainValDataset("val")
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batch_size, shuffle=True,drop_last=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of valing samples: %d\n" % int(len(dataset_val)))
    # Build model
    model = Net()
    print_network(model)
    # loss function
    criterion1 = nn.MSELoss(size_average=True)
    criterion = SSIM()
    
    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()
        criterion1 = nn.MSELoss(size_average=True).cuda()
        l1_loss = torch.nn.SmoothL1Loss().cuda()
       
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)  # learning rates
    # record training
    writer = SummaryWriter(opt.save_path)
    # load the lastest model
    # initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    # if initial_epoch > 0:
    #     print('resuming by loading epoch %d' % initial_epoch)
    #     model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    initial_step = findLastCheckpoint_step(save_dir=opt.save_path)
    if initial_step > 0:
        print('resuming by loading step %d' % initial_step)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_step%d.pth' % initial_step)))
    # start training

    step = initial_step
    p=0
    train_loss_sum=0
    for epoch in range(opt.epochs):
        scheduler.step(step)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input, target) in enumerate(loader_train, 0):
            # training step
            
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            if opt.use_gpu:
                input_train, target_train = Variable(input.cuda()), Variable(target.cuda())
            else:
                input_train, target_train = Variable(input), Variable(target)

            out_train= model(input_train)

            loss_mse = criterion1(out_train,target_train)
            loss_ssim = criterion(out_train,target_train)
            loss=loss_mse+0.2*(1-loss_ssim)
           
          
            loss.backward()
            optimizer.step()
            if i % 2== 0:
                p = p + 1
                train_loss_sum = train_loss_sum + loss.item()
            # training curve

           
            
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f,PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))

            
            model.eval()

            # log the images
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)

            if step % opt.iter_epoch == 0:
                # Log the scalar values
                epoch_loss = train_loss_sum / p
                writer.add_scalar('epoch_loss', epoch_loss, step)
                p = 0
                train_loss_sum = 0
            if step % opt.save_freq == 0:
               torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
               torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_step%d.pth' % (step)))

            step += 1

if __name__ == "__main__":
    main()
