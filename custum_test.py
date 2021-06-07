import argparse
import csv
import os
import random
import time
import warnings
import pandas as pd
import numpy as np
from itertools import chain
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from tensorboardX import SummaryWriter
from train_utils.utils import *
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100, CUSTUMCIFAR100
from train_utils.losses import LDAMLoss, FocalLoss

idx_to_label = {0: 'dolphin', 1: 'turtle', 2: 'lamp', 3: 'palm_tree', 4: 'castle', 5: 'bed', 6: 'maple_tree',
                7: 'orange', 8: 'sweet_pepper', 9: 'orchid', 10: 'lawn_mower', 11: 'seal', 12: 'train', 13: 'trout',
                14: 'bus', 15: 'spider', 16: 'chair', 17: 'worm', 18: 'crocodile', 19: 'television', 20: 'cup',
                21: 'mouse', 22: 'girl', 23: 'snail', 24: 'motorcycle', 25: 'keyboard', 26: 'tulip', 27: 'apple',
                28: 'road', 29: 'cloud', 30: 'squirrel', 31: 'wardrobe', 32: 'lion', 33: 'couch', 34: 'tank',
                35: 'sunflower', 36: 'camel', 37: 'bee', 38: 'telephone', 39: 'streetcar', 40: 'rabbit', 41: 'mountain',
                42: 'raccoon', 43: 'mushroom', 44: 'baby', 45: 'plain', 46: 'lizard', 47: 'skyscraper', 48: 'lobster',
                49: 'bicycle', 50: 'wolf', 51: 'rose', 52: 'leopard', 53: 'plate', 54: 'tiger', 55: 'table',
                56: 'caterpillar', 57: 'hamster', 58: 'woman', 59: 'cattle', 60: 'dinosaur', 61: 'possum', 62: 'otter',
                63: 'elephant', 64: 'shrew', 65: 'clock', 66: 'crab', 67: 'pickup_truck', 68: 'chimpanzee',
                69: 'aquarium_fish', 70: 'fox', 71: 'oak_tree', 72: 'willow_tree', 73: 'tractor', 74: 'bear',
                75: 'beaver', 76: 'beetle', 77: 'pear', 78: 'rocket', 79: 'man', 80: 'skunk', 81: 'butterfly',
                82: 'whale', 83: 'porcupine', 84: 'bridge', 85: 'bowl', 86: 'can', 87: 'snake', 88: 'boy',
                89: 'cockroach', 90: 'kangaroo', 91: 'sea', 92: 'house', 93: 'shark', 94: 'flatfish', 95: 'ray',
                96: 'forest', 97: 'poppy', 98: 'bottle', 99: 'pine_tree'}
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')
parser.add_argument('--data_root', default='./data', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_dir', type=str, default='custum')
parser.add_argument('--model_path', type=str,
                    default='/home/vision/jhkim/cifar_imbalance/checkpoint/custum/cifar100_resnet32_LDAM_DRW_exp_0.01_0/')


def main():
    args = parser.parse_args()
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=100, use_norm=True)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    model_path = os.path.join(args.model_path, 'ckpt.best.pth.tar')
    if os.path.exists(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    test_dataset = CUSTUMCIFAR100(root_dir=args.data_root, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    with torch.no_grad():
        files = []
        preds = []

        for i, (input, file) in enumerate(test_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            output = model(input)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t().squeeze()
            # print(len(file), pred.shape)
            files.append(list(file))
            preds.append(pred.cpu().tolist())
            # if i==3:
            #     break

    files = list(chain.from_iterable(files))
    files = [file.replace('100_lt', '100') for file in files]
    preds = list(chain.from_iterable(preds))
    preds = [idx_to_label[idx] for idx in preds]

    save_dir = os.path.join(os.getcwd(), args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data = {'id':files,
           'category':preds}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)

if __name__ == '__main__':
    main()
