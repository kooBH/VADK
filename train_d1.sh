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

## 2021-10-06
#TARGET=GPV_5
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  

## 2021-10-07
#TARGET=GPV_4
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  


## 2021-11-21
TARGET=DGD_1
python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  
