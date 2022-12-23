#!/bin/bash

#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

# 2021-09-29
#TARGET=GPV_5
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'

# 2021-03-13
TARGET=GPV_13
python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'
