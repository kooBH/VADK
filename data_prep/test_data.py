import torch

import librosa
import numpy as np

import os,glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

root_input  = '/home/data/kbh/AVTR/'
root_output = '/home/data/kbh/VADK/test/'

## TODO
list_target = [x ]



def process(idx):


if __name__ == '__main__' : 
    os.makedirs(root_output, exist_ok=True)

    num_cpu = cpu_count()
    arr = list(range(len(list_target)))
    with Pool(num_cpu) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='VADK : test data'))