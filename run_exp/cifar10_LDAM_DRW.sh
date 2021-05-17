#!/bin/bash


echo '=============================================================================================='
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
echo '=============================================================================================='

python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW

echo '=============================================================================================='
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW
echo '=============================================================================================='
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW 


echo '=============================================================================================='
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW
echo '=============================================================================================='
python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type LDAM --train_rule DRW


echo '=============================================================================================='
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW
echo '=============================================================================================='

python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type LDAM --train_rule DRW


