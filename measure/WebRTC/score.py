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

def score(idx,threshold):
    target_name = target_list[idx].split('/')[-1]
    target_id = target_name.split('.')[0]

    #print(target_name)

    vad = np.fromfile(target_list[idx],'<f4')

    label = scipy.io.loadmat(label_root+target_id+'.mat')['label'][0,:]

    #print(vad)
    #print(label)
    #print(np.shape(vad))
    #print(np.shape(label))

    fpr,tpr, thresholds = sklearn.metrics.roc_curve(label,vad) 
    auc = sklearn.metrics.auc(fpr, tpr)

    vad_flag = vad > threshold

    f1 = sklearn.metrics.f1_score(label,vad_flag)
    acc = sklearn.metrics.accuracy_score(label,vad_flag)

    return auc,f1,acc

#    print(thresholds)
    #print( str(tpr) +' | ' + str(fpr) +' | ' +str(auc))

def get_score(threshold):
    auc = 0
    f1 = 0
    acc = 0
    for i in range(len(target_list)):
        t1,t2,t3 = score(i,threshold)
        auc += t1
        f1 += t2
        acc += t3

    auc = auc/len(target_list)
    f1= f1/len(target_list)
    acc = acc/len(target_list)

    #print('--- treshold : ' + str(threshold) + ' ---- ')
    #print('AUC of ROC : '+ str(auc))
    #print('f1_score   : '+ str(f1))
    #print('accuracy   : '+str(acc))

    print(str(threshold) + ',' + str(auc) + ',' + str(f1) +','+ str(acc))


if __name__ == '__main__':
    cpu_num = cpu_count()
    cpu_num = 8

    os.makedirs(output_root,exist_ok=True)

    print('treshold,AUC,f1_score,accuracy')

    for i in range(10) :
        get_score(i/10)



#    arr = list(range(len(target_list)))
#    with Pool(cpu_num) as p:
#        r = list(tqdm(p.imap(score, arr), total=len(arr),ascii=True,desc='scoring'))




