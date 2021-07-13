import os,glob  

import librosa
import soundfile as sf
import numpy as np

from tqdm import tqdm


## keyboard noise
#data_root = '/home/data/kbh/AVTR/keyboard/'
#output_root = '/home/data/kbh/AVTR/keyboard/'

## hospital noise
data_root = '/home/data/kbh/AVTR/hospital_noise/'
output_root = '/home/data/kbh/AVTR/hospital_noise/'

target_list = [x for x in glob.glob(os.path.join(data_root,'*.wav'))]

data = None

for path in tqdm(target_list) : 
    raw,_ = librosa.load(path,sr=16000)
    raw = raw/np.abs(np.max(raw))

    data = np.append(data,raw)

print(np.shape(data))
print(type(data))
#sf.write(output_root + '/' + 'merged_keyboard.wav',data.astype(np.float64),16000)
sf.write(output_root + '/' + 'merged_hospial.wav',data.astype(np.float64),16000)