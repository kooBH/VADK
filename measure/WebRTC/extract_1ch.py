import os,glob
import numpy as np
import librosa
import soundfile as sf

# utils
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

audio_root = '/home/data/kbh/AVTR/raw/'
output_root = '/home/data/kbh/AVTR/labeled_1ch/'


labeled_file = open('/home/data/kbh/AVTR/list_labeled_wav.txt','r')
labeled_list = labeled_file.readlines()
# e.g. '100_200629-1734_홍0주.wav\n'  
len_labeled = len(labeled_list)

def measure(idx):
    target_name = labeled_list[idx][0:-1] # elim '\n'
    target_path = audio_root + target_name
    raw,_ = librosa.load(target_path,sr=16000)
    data = librosa.util.normalize(raw)
    sf.write(output_root+target_name,data,samplerate=16000)


    
if __name__ == '__main__':
    cpu_num = cpu_count()
    cpu_num = 8

    os.makedirs(output_root,exist_ok=True)

    arr = list(range(len(labeled_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(measure, arr), total=len(arr),ascii=True,desc='extract_1ch'))

