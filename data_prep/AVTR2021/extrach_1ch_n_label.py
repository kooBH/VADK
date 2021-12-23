import os,glob
import os.path
import numpy as np
import librosa
import torch
import soundfile as sf

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import json  

path_json = './label.private'
root_output = '/home/data/kbh/AVTR/extract/'
root_wav =  '/home/nas/DB/[DB]AV-TR/raw/'
sr = 16000

file_label = open("label.private")
json_label = json.load(file_label)
list_label = list(json_label.keys())

def process(idx):
    key_target = list_label[idx]
    id_target = key_target.split('_')[0]
    
    ## Wav
    path_wav = os.path.join(root_wav, key_target+'.wav')
    if os.path.isfile(path_wav):
        # load in Mono
        raw,_ = librosa.load(path_wav,sr=sr,mono=False)
        raw = raw[0,:]
        # Don't know librosa.load does normalzation
        data = librosa.util.normalize(raw)
        sf.write(os.path.join(root_output,'wav',id_target+'.wav'),data,samplerate=16000)
    else :
        return
    ## Label
    label_nurse   = json_label[key_target]['nurse']
    label_patient = json_label[key_target]['patient']
    label = label_nurse + label_patient

    label = [t.split('~') for t in label]

    y= [[float(z[0]),float(z[1])] for z in label]
    y.sort(key = lambda z: z[0]) 
    for i in range(len(y)) :
        for j in range(i+1,len(y)):
            if y[j][0] <= y[i][1] : 
                y[i][1] = y[j][1]
    label_mergred = []
    for i in range(len(y)-1) :
        j = len(y)-1-i
        if y[j][1] != y[j-1][1] :
            label_mergred.append(y[j])
    label_mergred.append(y[0])
    label_mergred.sort(key = lambda t: t[0]) 

    npy_label = np.array(label_mergred)
    np.save(os.path.join(root_output,'label',id_target+'.npy'),npy_label)
    

if __name__ == '__main__' : 

    cpu_num = cpu_count() 

    os.makedirs(root_output,exist_ok=True)
    os.makedirs(root_output+'/wav',exist_ok=True)
    os.makedirs(root_output+'/label',exist_ok=True)

    arr = list(range(len(list_label)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='AV-TR::extract 1ch & label'))

 
