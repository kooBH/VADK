import os,glob
import numpy as np
import scipy.io
import sklearn.metrics
import librosa
import torch

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sr      = 16000
n_fft   = 640
n_shift = 160
n_mels  = 64

n_frame = 960
n_frame_shift = 480

root       = '/home/data/kbh/'
root_label = root + '/AVTR/vad_label/'
root_data  = root + '/AVTR/labeled_1ch/'
root_output = root + '/VADK/AVTR/mel_'+str(n_mels)+'/'

list_target =  [ x for x in glob.glob(os.path.join(root_data,'*.wav'))]
print(len(list_target))

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

def process(idx):
    path_target = list_target[idx]

    name_target = path_target.split('/')[-1]
    id_target   = name_target.split('.')[0]
    id_target   = id_target.split('_')[0] + '_'+id_target.split('_')[1]

    path_label = root_label + id_target + '.mat'

    if os.path.exists(path_label) and os.path.exists(path_target): 
        pass
    else :
        return

    raw,_ = librosa.load(path_target,sr=sr)

    raw_label = scipy.io.loadmat(path_label)['label'][0,:]  

    # Wav to Mel
    spec = librosa.stft(raw,window='hann',n_fft=n_fft,hop_length=n_shift, win_length=None,center=True,dtype=np.cdouble)

    mel = np.matmul(mel_basis,np.abs(spec))
    pt = torch.from_numpy(mel)


    # synced label for stft
    label = np.zeros(len(raw_label))

    for i in range(len(raw_label)-3):
        if raw_label[i] or raw_label[i+1] or raw_label[i+2] or raw_label[i+3] :
            label[i]=1

    # save for 9600ms segments
    pt = pt[:,:-1]

    p_end = n_frame
    while p_end  <= pt.shape[1] : 
        t_pt = pt[:,p_end-n_frame:p_end]
        t_label = label[p_end-n_frame:p_end]

        #print(t_pt.shape)
        #print(t_label.shape)

        data = {"mel":t_pt,"label":t_label}
        torch.save(data,os.path.join(root_output,str(idx)+'_'+str(p_end)+'.pt'))

        p_end += n_frame_shift

    # print(pt.shape)
    # print(label.shape)

if __name__ == '__main__':
    cpu_num = cpu_count()

    os.makedirs(root_output,exist_ok=True)

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='AV-TR real data'))