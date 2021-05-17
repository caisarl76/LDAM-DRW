#!/bin/bash

echo '######################################################################'
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --dataset cifar100
echo '######################################################################'

# python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --dataset cifar100

echo '######################################################################'
echo python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --dataset cifar100
echo '######################################################################'

# python cifar_train.py --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --dataset cifar100


echo '######################################################################'
echo python cifar_train.py --gpu 0 --imb_type step --imb_factor 0.01 --loss_type CE --train_rule None --dataset cifar100
echo '######################################################################'

python cifar_train.py --gpu 0 --imb_type step --imb_factor 0.01 --loss_type CE --train_rule None --dataset cifar100

echo '######################################################################'
echo python cifar_train.py --gpu 0 --imb_type step --imb_factor 0.1 --loss_type CE --train_rule None --dataset cifar100
echo '######################################################################'

python cifar_train.py --gpu 0 --imb_type step --imb_factor 0.1 --loss_type CE --train_rule None --dataset cifar100
~

