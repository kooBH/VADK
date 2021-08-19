import os, glob
import torch
import numpy as np

class VAD_dataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True):
        if is_train : 
            self.list_path = [x for x in glob.glob(os.path.join(root,'train','*.pt'))]
        else :
            self.list_path = [x for x in glob.glob(os.path.join(root,'test','*.pt'))]
        print('dataset length : ' + str(len(self.list_path)))
        
    def __getitem__(self, index):
        path_item = self.list_path[index]

        # Process data_item if necessary.

        chunk = torch.load(path_item)
        # [n_mels, n_frame]
        img = chunk["mel"]
        label = chunk["label"]

        k = len(img[1]) % 32
        img = img[:, 0:-k]
        label = label[0:-k]

        img = np.reshape(img, (32, 32, -1))
        #label = np.reshape(label, (32, -1))

        img = np.transpose(img,(2,0,1))
        #label = np.transpose(label,(1,0))

        return img,label

    def __len__(self):
        return len(self.list_path)