import os, glob
import torch
import numpy as np

class VAD_dataset(torch.utils.data.Dataset):
    def __init__(self, hp,is_train=True) : 
        super(VAD_dataset,self).__init__()
        self.hp = hp
        root = hp.data.root
        self.nframe = hp.train.nframe
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

        if self.nframe > data["mel"].shape[1]:
            raise Exception("ERROR:: nframe is too large | " +str(self.nframe) +" > " + str(data["mel"].shape[1]))
        idx_start = np.random.randint(data["mel"].shape[1]-self.nframe)
        idx_end = idx_start + self.nframe

        data["mel"] = data["mel"][:,idx_start:idx_end]        
        data["mel"] = torch.unsqueeze(data["mel"],0)
        data["label"] = data["label"][idx_start:idx_end]
        
        if self.hp.model.specaug  and self.is_train:
            freq_l = np.random.randint(low=self.hp.specaug.freq_min,high=self.hp.specaug.freq_max)
            freq_s = np.random.randint(low=0,high=self.hp.model.n_mels-freq_l)
            data["mel"][0,freq_s:freq_s+freq_l,:] = 0

        shape = data["mel"].shape

        if 'd' in self.hp.model.input :
            d = torch.zeros(shape)
            # C, dim, T
            d[:,:-1,:] = data["mel"][0,1:,:]-data["mel"][0,0:-1,:]
            # channel-wise concat
            data["mel"] = torch.cat((data["mel"],d),0)
        
        if 'dd' in self.hp.model.input :
            dd = torch.zeros(shape)
            dd[:,:-2,:] = data["mel"][0,1:-1,:]-data["mel"][0,0:-2,:]
            # channel-wise concat
            data["mel"] = torch.cat((data["mel"],dd),0)

        #mel = data["mel"][:,idx_start:idx_end].float()
        #label = data["label"][idx_start:idx_end]
        data["label"] = torch.unsqueeze(data["label"],0)

        #if self.hp.model.label == 1:
        #    data["label"] = torch.unsqueeze(data["label"],0)
        #elif self.hp.model.label == 2:
        #    tmp_label = torch.zeros(2,data["label"].shape[0])
        #    tmp_label[0,:] = data["label"]
        #    tmp_label[1,:] = (~data["label"].bool()).float()
        #    data["label"] = tmp_label

        return data["mel"].float(),data["label"]
        #return mel,label

    def __len__(self):
        return len(self.list_path)