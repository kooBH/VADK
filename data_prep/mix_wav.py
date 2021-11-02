import os,glob
import numpy as np
import librosa
import torch 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import soundfile as sf

import warnings
warnings.filterwarnings('error')

root = '/home/data/kbh/'

root_output = root+'/VADK/test_official/'

#path_keyboard = '/home/data/kbh/AVTR/keyboard/merged_keyboard.wav'
root_keyboard = root+'/AVTR/Audioset_keyboard_valid/'
root_hospital = root+'/AVTR/hospital_16kHz/'
root_clean    = root+'/AVTR/AVA-Speech-clean/'
root_noise    = root+'/background_noise/'

n_output = 10000

sr      = 16000
n_fft   = 640
n_shift = 160
n_mels  = 32

sec_keyboard = 5.0
sec_hospital = 4.0
sec_clean    = 5.0
sec_noise    = 10.0

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

list_keyboard = [x for x in glob.glob(os.path.join(root_keyboard,'*.wav'))]
list_hospital = [x for x in glob.glob(os.path.join(root_hospital,'*.wav'))]
list_clean    = [x for x in glob.glob(os.path.join(root_clean,'*.wav'))]
list_noise    = [x for x in glob.glob(os.path.join(root_noise,'*.wav'))]

print(len(list_keyboard))
print(len(list_hospital))
print(len(list_clean))
print(len(list_noise))

def mix(idx):

    # random with multiprocessing
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    ## SNR keyboard [0,5]
    SNR_keyboard = np.random.rand(1)*5
    SNR_hospital = 5.0
    SNR_noise    = 5.0

    ##  
    path_keyboard = list_keyboard[np.random.randint(len(list_keyboard))]
    path_hospital = list_hospital[np.random.randint(len(list_hospital))]
    path_clean = list_clean[np.random.randint(len(list_clean))]
    path_noise = list_noise[np.random.randint(len(list_noise))]

    raw_keyboard,_  = librosa.load(path_keyboard,sr=sr)
    raw_hospital,_ = librosa.load(path_hospital,sr=sr)
    raw_clean,_     = librosa.load(path_clean,sr=sr)
    raw_noise,_     = librosa.load(path_noise,sr=sr)
    # Indexing
    n_sample_keyboard = int(sec_keyboard * sr)
    n_sample_hospital = int(sec_hospital * sr)
    n_sample_clean = int(sec_clean * sr)
    n_sample_noise = int(sec_noise * sr)

    idx_keyboard = np.random.randint(len(raw_keyboard) - n_sample_keyboard)
    idx_hospital = np.random.randint(len(raw_hospital) - n_sample_hospital)
    idx_clean    = np.random.randint(len(raw_clean)    - n_sample_clean)
    idx_noise    = np.random.randint(len(raw_noise)    - n_sample_noise)

    sample_keyboard = raw_keyboard[idx_keyboard : idx_keyboard + n_sample_keyboard]
    sample_hospital = raw_hospital[idx_hospital : idx_hospital + n_sample_hospital]
    sample_clean    = raw_clean   [idx_clean    : idx_clean    + n_sample_clean]
    sample_noise    = raw_noise   [idx_noise    : idx_noise    + n_sample_noise]

    # norm sample
    try : 
        norm_keyboard = sample_keyboard/np.max(np.abs(sample_keyboard))
        norm_hospital = sample_hospital/np.max(np.abs(sample_hospital))
        norm_clean    = sample_clean/np.max(np.abs(sample_clean))
        norm_noise    = sample_noise/np.max(np.abs(sample_noise))
    # there is invalid audio in hospital noise dataset
    except Warning : 
        print(path_hospital)
        exit()

    ## Mixing
    # SNR
    ratio_keyboard = np.power(10,SNR_keyboard/10)
    ratio_hospital = np.power(10,SNR_hospital/10)
    ratio_noise = np.power(10,SNR_noise/10)

    mean_energy_keyboard = np.sqrt(np.sum(np.power(norm_keyboard,2)))/n_sample_keyboard
    mean_energy_hospital = np.sqrt(np.sum(np.power(norm_hospital,2)))/n_sample_hospital
    mean_energy_clean = np.sqrt(np.sum(np.power(norm_clean,2)))/n_sample_clean
    mean_energy_noise = np.sqrt(np.sum(np.power(norm_noise,2)))/n_sample_noise

    energy_normal_keyboard = np.sqrt(mean_energy_clean)/np.sqrt(mean_energy_keyboard)
    energy_normal_hospital = np.sqrt(mean_energy_clean)/np.sqrt(mean_energy_hospital)
    energy_normal_noise = np.sqrt(mean_energy_clean)/np.sqrt(mean_energy_noise)

    weight_keyboard = energy_normal_keyboard/np.sqrt(ratio_keyboard)
    weight_hospital = energy_normal_hospital/np.sqrt(ratio_hospital)
    weight_noise    = energy_normal_noise/np.sqrt(ratio_noise)

    mix_keyboard = norm_keyboard * weight_keyboard
    mix_hospital = norm_hospital * weight_hospital
    mix_noise = norm_noise * weight_noise
    # Mix
    offset_keyboard = np.random.randint(n_sample_noise - n_sample_keyboard)
    offset_hospital = np.random.randint(n_sample_noise - n_sample_hospital)
    offset_clean    = np.random.randint(n_sample_noise - n_sample_clean)

    mixed                                                       = mix_noise
    mixed[offset_keyboard:offset_keyboard + n_sample_keyboard] += mix_keyboard
    mixed[offset_hospital:offset_hospital + n_sample_hospital] += mix_hospital
    mixed[offset_clean:offset_clean       + n_sample_clean]    += norm_clean
    mixed = mixed/np.max(np.abs(mixed))

    # Label (ceil on edge)
    label = torch.zeros(int(np.ceil(n_sample_noise/n_shift)) + 1-int(n_fft/n_shift) )
    label[int(np.ceil(offset_clean/n_shift)) :int(np.ceil((offset_clean+n_sample_clean)/n_shift)) ] = 1

    ## save save

    sf.write(root_output +'/wav/'+str(idx)+'.wav',mixed,16000)

    ## label .pt
    torch.save(label,root_output + '/label/'+str(idx)+'.pt')



if __name__=='__main__' : 
    cpu_num = cpu_count()

    print(cpu_num)

    os.makedirs(root_output,exist_ok=True)

    arr = list(range(n_output))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(mix, arr), total=len(arr),ascii=True,desc=''))

