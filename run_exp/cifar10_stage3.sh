#!/bin/bash

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 50 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 60 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 70 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 80 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 90 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3

python tail_as_head.py --imb_factor 0.1 --gpu 0 --t_as_h 0 --epoch 100 50 100 --root_log log/stage_3 --root_model checkpoint/stage_3





