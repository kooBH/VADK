import librosa
import pandas as pd
import numpy as np
import os,glob

import soundfile as sf
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

input_root =  '/home/data/kbh/AVTR/AVA-Speech-wav/'
output_root = '/home/data/kbh/AVTR/AVA-Speech-clean/'
sr = 16000

# csv
ava = pd.read_csv('ava_speech_labels_v1.csv',names=['id','start','end','label'])

# only id
series_id = ava['id'].drop_duplicates()
list_id = series_id.values

def extract(idx):
    target_id = list_id[idx]
    target_path = input_root +  target_id + '.wav'

    # check whether file is donwloaded well
    if os.path.isfile(target_path) :
        pass
    else : 
        return

    target_info = ava.loc[ava['id'] == target_id]

    raw,_ = librosa.load(target_path,sr=sr)

    concat = None
    for row in target_info.iterrows():
        if row[1]['label'] == 'CLEAN_SPEECH' : 
            start_idx = int((row[1]['start']-900)*sr)
            end_idx = int((row[1]['end']-900)*sr)
            if concat is None :
                concat = raw[start_idx:end_idx]
            else :
                concat = np.concatenate((concat,raw[start_idx:end_idx]))
    if concat is None :
        print(target_id)
        return
    # normalize
    concat = concat / np.abs(np.max(concat))
    sf.write(output_root + target_id + '.wav',concat,sr)

cpu_num = cpu_count()
os.makedirs(output_root,exist_ok=True)

arr = list(range(len(list_id)))
with Pool(cpu_num) as p:
    r = list(tqdm(p.imap(extract, arr), total=len(arr),ascii=True,desc='extract clean'))