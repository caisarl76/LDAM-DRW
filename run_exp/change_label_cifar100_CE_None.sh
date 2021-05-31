#!/bin/bash

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.1 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.1 --dataset cifar100



echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.2 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.2 --dataset cifar100

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.3 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.3 --dataset cifar100


echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.4 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.4 --dataset cifar100




echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.1 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.1 --dataset cifar100



echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.2 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.2 --dataset cifar100

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.3 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.3 --dataset cifar100


echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.4 --dataset cifar100
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.4 --dataset cifar100

