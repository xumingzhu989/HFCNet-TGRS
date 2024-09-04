import torch
import argparse
from torch import nn
from util import dataset, transform, config
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from util.util import check_makedirs
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from model import supervisionAtt
import torch.optim as optim
import datetime
import math
import os
import pytorch_iou
import cv2

eps = math.exp(-10)

def createModel(args):
    net = model.HFCNet(args)
    return net
import numpy as np
import random


# 解析参数
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/dataset_o.yaml', help='config file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_id', type=str, default='HFCNet')
    parser.add_argument('--flag', type=str, default='train')
    parser.add_argument('opts', help='see config/cod_mgl50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg, args


# 0~1 normalization.
def MaxMinNormalization(x):
    Max = torch.max(x)
    Min = torch.min(x)
    x = torch.div(torch.sub(x, Min), 0.0001 + torch.sub(Max, Min))
    return x

def flat(mask,h,w):
    batch_size = mask.shape[0]
    # print(mask.shape)
    mask = F.interpolate(mask,size=(int(h),int(w)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
    g = x @ x.transpose(-2,-1) # b hw hw
    return g

class MyLoss:
    def __init__(self):
        self.x = None
        self.y = None
        self.IOU = pytorch_iou.IOU(size_average=True)

    def loss(self, X, y, atts, f_s, f_c): 
        # print('x the size is ', x.shape)
        # print('y the size is ', y.shape)
        lossAll = 0
        for x in X:
            loss = (-y.mul(torch.log(x + eps)) - (1 - y).mul(torch.log(1 - x + eps))).sum()  
            num_pixel = y.numel()
            lossAll = lossAll + torch.div(loss, num_pixel) + self.IOU(x, y)
        # hw = [80,40,20]
        hw = [56,28,14]
        # hw = [28,14,7]
        # hw = [64,32,16]
        # hw = [72,36,18]
        i = 0
        for att,fs,fc in zip(atts,f_s,f_c):
            # f_s (B,1,H,W)
            h = hw[i]
            w = hw[i]
            i = i+1
            g = flat(y.unsqueeze(1),h,w)
            att = att.squeeze(1)
            lossAll = lossAll + F.binary_cross_entropy_with_logits(att, g) + F.mse_loss(fs,fc)

        return lossAll


# loss曲线
def loss_curve(counter, losses):
    fig = plt.figure()
    plt.plot(counter, losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number')
    plt.ylabel('loss')
    plt.show()  # 画出loss变化曲线


def train(loss_fn, args, arg):
    dev = arg.device
    model_id = arg.model_id
    # set_seed(1)
    train_losses = []
    train_counter = []

    # 训练数据转换及数据载入
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.RandomVerticalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    dataset_root = args.data_root
    if args.data_name == 'ORSSD':
        train_data = dataset.ORSSD(dataset_root, 'train', transform=train_transform)
    elif args.data_name == 'EORSSD':
        train_data = dataset.EORSSD(dataset_root, 'train', transform=train_transform)
    elif args.data_name == 'ORSI':
        train_data = dataset.EORSSD(dataset_root, 'train', transform=train_transform)
    else:
        train_data = None
    train_sampler = None
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    # 定义网络

    net = createModel(args)
    device = torch.device(dev if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0000,
    #                        amsgrad=False)
    optimizer = optim.Adam(model.parameters(), lr=0.00007, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0000,
                           amsgrad=False)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    date_str = str(datetime.datetime.now().date())  # 获取当前时间的日期
    # 训练模型
    model.train()
    lossAvg = 0
    interNum = 10
    # interNum = 30  # 120
    for epoch in range(start_epoch, args.epoch_num):
        for i, (input, gt, _) in enumerate(train_loader):
            input = input.to(device)
            gt = gt.to(device)
            optimizer.zero_grad()
            outs, atts , fs, fc= model(input)
            gt = MaxMinNormalization(gt)
            loss = loss_fn.loss(outs, gt, atts, fs, fc)
            # loss = loss_fn.loss(outs, gt)
            loss.backward()
            optimizer.step()
            if ((i + 1) % interNum == 0) or (i + 1 == len(train_loader)): 
                print('Train Epoch: {} [{:.0f}/{} ({:.0f}%)]\tLoss: {:.10f}'
                      .format(epoch + 1, 
                              (i + 1) * len(input),
                              len(train_data),
                              100. * (i + 1) * args.train_batch_size / len(train_data),
                              lossAvg / (len(train_data) / args.train_batch_size)))
            lossAvg = lossAvg + loss.item()
            train_losses.append(loss.item())
            train_counter.append(((epoch) * len(train_data) / args.train_batch_size) + i)
        lossAvg = 0
        print("第%d个epoch的学习率：%f" % (epoch + 1, optimizer.param_groups[0]['lr']))
        scheduler.step()
        if ((epoch + 1) % 1 == 0) or (epoch + 1 == args.epoch_num):
            model_folder = args.model_path + date_str + '/'
            check_makedirs(model_folder)
            save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
            # if epoch>34:
            torch.save(save_file, model_folder +''+ model_id+"_"+str(epoch+1) + '.pth')
    loss_curve(train_counter, train_losses)


def test(args, arg):
    # 测试数据转换及数据载入
    dev = arg.device
    model_id = arg.model_id
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    test_transform = transform.Compose([
        transform.Resize((args.img_h, args.img_w)),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    to_pil_img = transforms.ToPILImage()  # 将输出的tensor转换为PIL image

    date_str = str(datetime.datetime.now().date())  # 获取当前时间的日期
    results_folder = args.results_folder + date_str  # '2024-08-01/'  # 以当前日期命名文件夹

    if args.data_name == 'ORSSD':
        test_data = dataset.ORSSD(args.data_root, 'test', transform=test_transform)
    elif args.data_name == 'EORSSD':
        test_data = dataset.EORSSD(args.data_root, 'test', transform=test_transform)
    elif args.data_name == 'ORSI':
        test_data = dataset.EORSSD(args.data_root, 'test', transform=test_transform)
    else:
        test_data = None
    img_path_list = test_data.image_paths
    img_name_list = []
    n_imgs = len(test_data)
    for i in range(n_imgs):
        img_name = img_path_list[i].split('/')[
            -1]  # img_path_list[i][0] is the image path, img_path_list[i][1] is the gt path
        img_name_list.append(img_name)

    test_sampler = None
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True,
                             sampler=test_sampler,
                             drop_last=True)

    epoch_list = [
        40
        ]
    for now in epoch_list:
        net = createModel(args)
        device = torch.device(dev if torch.cuda.is_available() else 'cpu')
        model = net.to(device)
        model_dir = args.model_path + '2024-07-17/' + ''+model_id+'_' + str(
            now) + ".pth"  # the path/file to save the trained model params.
        results_folder_now = results_folder + "/" + ''+model_id+'_' + str(now)
        model.load_state_dict(torch.load(model_dir)['model'])
        print('The network parameters are loaded!')

        # 测试模型
        model.eval()
        for i, (input, _, img_size) in enumerate(test_loader):
            input = input.to(device)
            sal1 = model(input)[0][0]
            # sal1, sal2, sal3, sal4, sal5= model(input)
            # sal1, sal2, sal3, sal4 = model(input)
            n_img, _, _ = sal1.size()
            for j in range(n_img):
                salmaps = to_pil_img(sal1[j].cpu())
                salmaps = salmaps.resize((int(img_size[j][1]), int(img_size[j][0])))  # PIL.resize(width, height)
                file_name = img_name_list[i * args.test_batch_size + j]  # get the corresponding image name.
                # salmaps.show()
                print(file_name)
                the_name = file_name.split('.')[0]
                file_name = the_name+'.png'
                # os.system("pause")
                if not os.path.isdir(results_folder_now):
                    os.makedirs(results_folder_now)
                salmaps.save(results_folder_now + '/' + file_name)
                print('Testing {} th image'.format(i * args.test_batch_size + j))


if __name__ == '__main__':

    args, arg = get_parser()
    Flag_train_test = arg.flag 
    print(Flag_train_test)
    if Flag_train_test == 'train':
        criterion = MyLoss()
        train(loss_fn=criterion, args=args, arg=arg)
    else:
        ##########################################################################################
        # to test the network.
        test(args=args, arg=arg)
