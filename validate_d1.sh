#!/bin/bash

TARGET=GPV_4
python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt
#TARGET=miso_3
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

#TARGET=miso_1_ver2
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

#TARGET=GPV_1
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

#TARGET=miso64_2
#python src/validation.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'  --chkpt /home/nas/user/kbh/VADK/chkpt/${TARGET}/bestmodel.pt

