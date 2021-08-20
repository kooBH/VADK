#!/bin/bash

TARGET=v1
python src/trainer.py  -c ./config/${TARGET}.yaml -v ${TARGET}  --device 'cuda:1'