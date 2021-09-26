import numpy
import os,glob
import shutil

ratio_test = 0.1
root = '/home/data2/kbh/VADK/AVTR/mel40'

list_target = [x for x in glob.glob(os.path.join(root,'*.pt'))]

if __name__ == '__main__' :
    os.makedirs(root+'/train',exist_ok=True)
    os.makedirs(root+'/test',exist_ok=True)


    list_idx_test = numpy.random.choice(len(list_target),int(len(list_target)*ratio_test),replace=False)

    for idx in list_idx_test:
        shutil.move(list_target[idx],root+'/test')

    list_train = [x for x in glob.glob(os.path.join(root,'*.pt'))]
    for path in list_train:
        shutil.move(path,root+'/train')
