import argparse
import random
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

import models
from train_utils.train_utils import *
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

head_to_class = {'cifar10': {8: 0}, 'cifar100': {78: 18, 79: 44, 88: 50, 89: 8, 98: 2, 99: 61}}

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet32)')
parser.add_argument('--t_as_h', default=0, type=int)
#### t_as_h
# -1: not change label
# 0 : change label tail to head
# 1 : change label head to tail
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--root_log', type=str, default='log/t_as_h')
parser.add_argument('--root_model', type=str, default='checkpoint/t_as_h')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, 'CE', 'None', 'exp', str(args.imb_factor), args.exp_str, str(args.t_as_h)])
    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def stage2(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=False)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    cudnn.benchmark = True

    # Data loading code

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.rand_number, train=True, download=True, t_as_h=args.t_as_h)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                          rand_number=args.rand_number, train=True, download=True, t_as_h=args.t_as_h)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'stage1/log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'stage1/log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'stage1/args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name, 'stage1'))

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    ########################################### STAGE 2 ##################################################
    return
    print('')
    print('########################################### STAGE 2 ##################################################')
    print('')
    if args.dataset == 'cifar10':
        train_dataset_2 = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                           rand_number=args.rand_number, train=True, download=True)
    elif args.dataset == 'cifar100':
        train_dataset_2 = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                            rand_number=args.rand_number, train=True, download=True)
    else:
        warnings.warn('Dataset is not listed')
        return

    train_loader = torch.utils.data.DataLoader(train_dataset_2, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=args.workers,
                                             pin_memory=True)

    torch.nn.init.xavier_uniform_(model.linear.weight)
    optimizer = torch.optim.SGD(model.parameters(), args.lr * 0.5,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'stage2/log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'stage2/log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'stage2/args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name, 'stage2'))


    for epoch in range(50):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    main()
