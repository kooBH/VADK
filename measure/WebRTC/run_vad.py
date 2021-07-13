import os,glob
import numpy as np
import librosa
import scipy.io

# utils
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

label_root = '/home/data/kbh/AVTR/json2mat/'
audio_root = '/home/data/kbh/AVTR/labeled_1ch/'
output_root = '/home/data/kbh/AVTR/WebRTC/vad_result/'
rnnvad_path = '/home/kbh/shared_work/VADK/webrtc_rnnvad/bin/webrtc_rnnvad'


## WebRTC VAD 
## - output : 10ms segments speech probability

target_list = [x for x in glob.glob(os.path.join(audio_root,'*.wav'))]

# e.g.
# file name    : 39_200427-1042_김0순.wav.wav
# labels names : 39_200427-1042_nurse.mat
#                39_200427-1042_patient.mat


def rnnvad(idx):
    target_name = target_list[idx].split('/')[-1]
    tmp_str = target_name.split('_')
    target_id = tmp_str[0]+'_'+tmp_str[1]

    ## run rnnvad
    subprocess.run([rnnvad_path +" -i "+target_list[idx]+" -o "+ output_root +target_id+".bin"],shell=True)

if __name__ == '__main__':
    cpu_num = cpu_count()

    arr = list(range(len(target_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(rnnvad, arr), total=len(arr),ascii=True,desc='WebRTC RNN-VAD'))

