#!/bin/bash

#TARGET=GPV_2
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt
#TARGET=GPV_3
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt
#TARGET=GPV_9
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt


TARGET=DGD_1
python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:0'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt
