#!/bin/bash

echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.1
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.1


echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.2
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.2

echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.3
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.3


echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.4
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.4



echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.1
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.01_0.pickle --change-portion 0.1

echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.2
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.01_0.pickle --change-portion 0.2


echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.3
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.01_0.pickle --change-portion 0.3

echo '=============================================================================================='
echo python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.1_0.pickle --change-portion 0.4
echo '=============================================================================================='

python train_transferred_dataset.py --gpu 1 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --t_h_file data/new_cifar10_resnet32_CE_None_exp_0.01_0.pickle --change-portion 0.4


