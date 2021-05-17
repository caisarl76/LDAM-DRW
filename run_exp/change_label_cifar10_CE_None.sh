#!/bin/bash

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.1
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.1



echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.2
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.2

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.3
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.3


echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.4
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.01 --loss_type CE --train_rule None --switch-prob 0.4




echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.1
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.1



echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.2
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.2

echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.3
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.3


echo '=============================================================================================='
echo python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.4
echo '=============================================================================================='
python train_with_cofusion_matrix.py  --gpu 0 --imb_type exp --imb_factor 0.1 --loss_type CE --train_rule None --switch-prob 0.4

