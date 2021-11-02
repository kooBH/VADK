#!/bin/bash

#TARGET=GPV32_1
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1' --chkpt 'bestmodel.pt'
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  

## 2021-09-26
#TARGET=GPV_2
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  

## 2021-09-28
#TARGET=miso_3
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  

## 2021-09-29
#TARGET=miso_4
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  
TARGET=GPV_6
python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  
