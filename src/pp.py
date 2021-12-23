import numpy as np




def label_post_processing(label,range_smoothing=1):
    label_pp = np.zeros(np.shape(label))

    ## smoothing ?
    flag = False
    cnt = 0
    label_pp = label.coplabel_pp()
    for idlabel in range(len(label)):
        if label[idlabel] == 1 :
            flag = True
            cnt = 2
        else :
            # smoothing
            if cnt != 0 :
                label_pp[idlabel]=1
                cnt-=1
            else :
                flag = False


    return label_pp