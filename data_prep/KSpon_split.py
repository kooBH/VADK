import os,glob 
import tqdm

import librosa
import soundfile as sf
import numpy as np

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

threshold_dB = 30
shift = 1024
frame = 1024*4

data_root = '/home/data/kbh/AVTR/Kspon_WAV'
output_root = '/home/data/kbh/AVTR/Kspon_split'

target_list = [x for x in glob.glob(os.path.join(data_root, '*.wav'))]

def split(idx):
    target_path = target_list[idx]

    target_name = target_path.split('/')[-1]
    target_name = target_name.split('.')[0]

    raw,_ = librosa.load(target_path,sr=16000)

    intervals = librosa.effects.split(raw,top_db = threshold_dB, frame_length=frame, hop_length=shift)

    for i in range(len(intervals)) : 

        # to split by silent intervals
        split = raw[intervals[i,0]:intervals[i,1] ]

        # Exception
        if not len(split) > 0 :
            continue
        if len(split) < frame : 
            continue

        # normalization
        split = split/np.abs(np.max(split))
    
        sf.write( output_root+'/'+target_name+'_'+ str(i)+'.wav' ,split ,16000) 

os.makedirs(output_root,exist_ok=True)

cpu_num = cpu_count()

arr = list(range(len(target_list)))
with Pool(cpu_num) as p:
    r = list(tqdm(p.imap(split, arr), total=len(arr),ascii=True,desc='split'))