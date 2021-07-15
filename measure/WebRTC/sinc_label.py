import os,glob
import numpy as np
import librosa
import scipy.io

# utils
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

label_root = '/home/data/kbh/AVTR/json2mat/'
label_output_root = '/home/data/kbh/AVTR/WebRTC/vad_label/'
rnnvad_root= '/home/data/kbh/AVTR/WebRTC/vad_result/'

## WebRTC VAD 
## - output : 10ms segments speech probability

target_list = [x for x in glob.glob(os.path.join(rnnvad_root,'*.bin'))]
threshold_list = []
len_target = len(target_list)

score = {''}

def measure(idx):
    target_name = target_list[idx].split('/')[-1]
    target_id = target_name.split('.')[0]

    #print(target_id)

    label_nurse_name = target_id + '_nurse.mat'
    label_patient_name = target_id + '_patient.mat'

    ## check label's existence
    if (not os.path.isfile(os.path.join(label_root,label_nurse_name))) or (not os.path.isfile(os.path.join(label_root,label_patient_name)) ) :
        return

    ## load vad result

    ## '>f4' : big-endian float32
    ## '<f4' : little-endian float32
    data = np.fromfile(target_list[idx],'<f4')

    ## load label
    label_nurse   = scipy.io.loadmat(label_root+label_nurse_name)['nurse_time_mat']
    label_patient = scipy.io.loadmat(label_root+label_patient_name)['patient_time_mat']

    ## WebRTC RNNVAD segment is 10ms.
    def trim(x):
        # floor onset
        x[:,0] = x[:,0]*100
        x[:,0] = np.floor(x[:,0])
        x[:,0] = x[:,0]/100
        # ceil offset
        x[:,1] = x[:,1]*100
        x[:,1] = np.floor(x[:,1])
        x[:,1] = x[:,1]/100
        return x
    
    label_nurse =  trim(label_nurse)
    label_patient = trim(label_patient)

    ## merge nurse and patient
    label = np.zeros(len(data))
    #print(len(label))

    for i in label_nurse:
        #print(str(int(i[0]*100)) + ' |' + str(int(i[1]*100)))
        label[int(i[0]*100):int(i[1]*100)]=1
    for i in label_patient:
        #print(str(int(i[0]*100)) + ' |' + str(int(i[1]*100)))
        label[int(i[0]*100):int(i[1]*100)]=1

    #print(np.sum(label))
    scipy.io.savemat(label_output_root+target_id+'.mat',{"label":label})
    
     
if __name__ == '__main__':
    cpu_num = cpu_count()

    print(len(target_list))

    arr = list(range(len(target_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(measure, arr), total=len(arr),ascii=True,desc='sinc_label'))

