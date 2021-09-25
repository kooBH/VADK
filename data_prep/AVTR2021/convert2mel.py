import os,glob
import numpy as np
import librosa
import torch

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sr      = 16000
n_fft   = 512
n_shift = 128
n_overlap = n_fft - n_shift
n_mels  = 40
time_shift = (1/sr)*n_shift

root_input = '/home/data2/kbh/AVTR/extract/'
root_output = '/home/data2/kbh/AVTR/'

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# split into train / test 
list_data = [x for x in glob.glob(os.path.join(root_input,'wav','*.wav'))]

def process(idx):
    path_target = list_data[idx]
    id_data = path_target.split('/')[-1]
    id_data = id_data.split('.')[0]
    path_label = os.path.join(root_input,'label',id_data+'.npy')
   
    ## Load
    npy_label = np.load(path_label)
    raw,_     = librosa.load(path_target,sr=sr)

    ## Data
    spec = librosa.stft(raw,window='hann',n_fft=n_fft,hop_length=n_shift, win_length=None,center=True,dtype=np.cdouble)
    mel = np.matmul(mel_basis,np.abs(spec))
    pt = torch.from_numpy(mel)
    pt = pt.float()
    
    # Label
    
    raw_label = np.zeros(np.shape(mel)[1])
    label = np.zeros(np.shape(raw_label))
    
    for t_label in npy_label :
        idx_start = int(t_label[0]/time_shift)
        idx_end = int(t_label[1]/time_shift)
        #print(str(idx_start) + ' | ' + str(idx_end))
        raw_label[idx_start:idx_end]=1
    
    # if single shift in frame is activate that block is active
    for i in range(len(raw_label)-3):
        if raw_label[i] or raw_label[i+1] or raw_label[i+2] or raw_label[i+3] :
            label[i]=1
    label = torch.from_numpy(label)
    label = label.float()
    data = {"mel":pt, "label":label}
    #print(np.shape(raw))
    #print(pt.shape)
    #print(label.shape)
        
    ## save
    torch.save(data, os.path.join(root_output,'mel'+str(n_mels),id_data+'.pt'))


if __name__ == '__main__' : 
    cpu_num = cpu_count() - 14

    os.makedirs(root_output+'/mel'+str(n_mels),exist_ok=True)

    arr = list(range(len(list_data)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='AV-TR::convert to mel '+str(n_mels)))
        


