#!/bin/bash

#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

# 2021-09-27
#TARGET=GPV_2
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'


# 2021-09-28
#TARGET=GPV_3
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'

# 2021-10-06
#TARGET=GPV_2
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'

# 2021-11-03
#TARGET=GPV_7
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'


# 2021-11-19
#TARGET=GPV_8
#python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'

# 2021-11-21
TARGET=GPV_9
python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'
