import os, glob
import torch
import numpy as np

class VAD_dataset(torch.utils.data.Dataset):
    def __init__(self, root,is_train=True) : 
        if is_train : 
            self.list_path = [x for x in glob.glob(os.path.join(root,'train','*.pt'))]
        else :
            self.list_path = [x for x in glob.glob(os.path.join(root,'test','*.pt'))]
        print('dataset length : ' + str(len(self.list_path)))

    def __gettime__(self,index):
        path_item = self.list_path[index]

        # data["mel"] :
        # - [n_mels, n_frame] = [32, 960]
        # - 10ms shift 75% overlap
        # data["label"] :
        # - binary label
        data = torch.load(path_item)
        return data["mel"].float(),data["label"]

    def __len__(self):
        return len(self.list_path)