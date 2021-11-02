import os, glob
import torch
import numpy as np

class VAD_dataset(torch.utils.data.Dataset):
    def __init__(self, hp,is_train=True) : 
        super(VAD_dataset,self).__init__()
        root = hp.data.root
        #self.nframe = hp.train.nframe
        self.is_train = is_train

        if is_train : 
            self.list_path = [x for x in glob.glob(os.path.join(root,'train','*.pt'))]
        else :
            self.list_path = [x for x in glob.glob(os.path.join(root,'test','*.pt'))]

        self.true_len = len(self.list_path)

    def __getitem__(self,index):
        path_item = self.list_path[index]

        # data["mel"] :
        # - [n_mels, n_frame] = [32, 960]
        # - 10ms shift 75% overlap
        # data["label"] :
        # - binary label
        data = torch.load(path_item)
        #idx_start = np.random.randint(data["mel"].shape[1]-self.nframe)
        #idx_end = idx_start + self.nframe

        #mel = data["mel"][:,idx_start:idx_end].float()
        #label = data["label"][idx_start:idx_end]

        return data["mel"].float(),data["label"]
        #return mel,label

    def __len__(self):
        return len(self.list_path)