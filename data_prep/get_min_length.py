import os,glob 

import librosa
import numpy as np

data_root = '/home/data/kbh/AVTR/Kspon_split'

target_list = [x for x in glob.glob(os.path.join(data_root, '*.wav'))]


min = 1000000000
avg = 0
num_under_2048 = 0
num_under_100 = 0

print(len(target_list))

for target_path in target_list : 
    raw,_ = librosa.load(target_path,sr=16000)

    #print( target_path+' : '+ str(len(raw)))

    avg += len(raw)
    if len(raw) < 2048 : 
        num_under_2048 +=1

    if len(raw) < 100 : 
        num_under_100 +=1

    if len(raw) < min :
        min = len(raw)
    

avg = avg/len(target_list)

print('minimum length of wav files in ' + data_root +' is ')
print(min)
print('average length of wav files in ' + data_root +' is ')
print(avg)
print('num under 2048')
print(num_under_2048)
print('num under 100')
print(num_under_100)
