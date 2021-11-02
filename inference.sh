#!/bin/bash

#python ./src/inference.py -i <input dir> -o <output dir> -c <config> -d <device> -n <num_process> -p <model path>
python ./src/inference.py -i /home/nas/user/kbh/VADK/test_dir/ -o /home/nas/user/kbh/VADK/wav_output -c ./config/GPV_2.yaml -d cuda:0 -n 4 -p /home/nas/user/kbh/VADK/chkpt/GPV_2/bestmodel.pt