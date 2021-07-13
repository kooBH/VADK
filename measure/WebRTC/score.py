import os,glob
import numpy as np
import scipy.io
import sklearn.metrics

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

root = '/home/data/kbh/AVTR/WebRTC/'
label_root = '/home/data/kbh/AVTR/WebRTC/vad_label/'
vad_root = '/home/data/kbh/AVTR/WebRTC/vad_result/'
output_root = root + 'score/'


target_list = [x for x in glob.glob(os.path.join(vad_root,'*.bin'))]

def score(idx):
    target_name = target_list[idx].split('/')[-1]
    target_id = target_name.split('.')[0]

    vad = np.fromfile(target_list[idx],'<f4')

    label = scipy.io.loadmat(label_root+target_id+'.mat')['label'][0,:]

    #print(vad)
    #print(label)
    #print(np.shape(vad))
    #print(np.shape(label))

    fpr,tpr, thresholds = sklearn.metrics.roc_curve(label,vad) 
    auc = sklearn.metrics.auc(fpr, tpr)

#    print(thresholds)
    print( str(tpr) +' | ' + str(fpr) +' | ' +str(auc))



if __name__ == '__main__':
    cpu_num = cpu_count()
    cpu_num = 8

    os.makedirs(output_root,exist_ok=True)

    for i in range(len(target_list)):
        score(i)
    exit()

    arr = list(range(len(target_list)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(score, arr), total=len(arr),ascii=True,desc='scoring'))




