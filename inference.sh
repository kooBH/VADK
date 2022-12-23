#!/bin/bash

#python ./src/inference.py -i <input dir> -o <output dir> -c <config> -d <device> -n <num_process> -p <model path>
#python ./src/inference.py -i /home/nas/user/kbh/VADK/test_dir/ -o /home/nas/user/kbh/VADK/wav_output -c ./config/GPV_2.yaml -d cuda:0 -n 4 -p /home/nas/user/kbh/VADK/chkpt/GPV_2/bestmodel.pt

#python ./src/inference.py -i '/home/nas/DB/\[DB\]AV-TR/\[20211102\]\ AVCV 시뮬레이션(도희준형)/' -o /home/nas/user/kbh/VADK/wav_output -c ./config/GPV_2.yaml -d cuda:0 -n 4 -p /home/nas/user/kbh/VADK/chkpt/GPV_2/bestmodel.pt
#python ./src/inference.py -i '/home/kbh/shared_work/VADK/sample' -o './out' -c ./config/GPV_2.yaml -d cuda:0 -n 1 -p /home/nas/user/kbh/VADK/chkpt/GPV_2/bestmodel.pt -t 0.7


#TARGET=GPV_5
#python ./src/inference.py -i '/home/nas/user/kbh/uma8' -o /home/nas/user/kbh/VADK/uma8/${TARGET} -c ./config/${TARGET}.yaml -d cuda:0 -n 4 -p /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt -t 0.5
#TARGET=GPV_10
#python ./src/inference.py -i '/home/nas/user/kbh/uma8' -o /home/nas/user/kbh/VADK/uma8/${TARGET}_u -c ./config/${TARGET}.yaml -d cuda:0 -n 1 -p /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt -t 0.5 -u 50
#TARGET=GPV_11
#python ./src/inference.py -i '/home/nas/user/kbh/uma8' -o /home/nas/user/kbh/VADK/uma8/${TARGET} -c ./config/${TARGET}.yaml -d cuda:0 -n 1 -p /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt -t 0.5 -u 50


TARGET=GPV_12
python ./src/inference.py -i sample -o output -c ./config/${TARGET}.yaml -d cuda:0 -n 1 -p /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt -t 0.5 -u 50
#TARGET=GPV_11
