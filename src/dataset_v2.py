import os, glob
import torch
import numpy as np

class dataset_v2(torch.utils.data.Dataset):
    def __init__(self, 
        root,
        kind_feature=["mel"],
        fs = 16000,
        n_fft = 512,
        n_hop= 128,
        n_frame = 600,
        n_mels = 40,
        specaug=True
        ) : 
        super(dataset_v2,self).__init__()

        # PATH
        self.root = root
        self.list_wav   = glob.glob(os.path.join(root,"*.wav"))

        # PARAM
        self.fs = fs
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.n_frame = n_frame
        self.time_shift = (1/self.fs)*n_hop

        # PROCESS
        self.n_mels = n_mels
        self.mel_basis = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels)

    def __getitem__(self,index):
        path_wav = self.list_data[index]
        name_wav = path_wav.split('/')[-1]
        id_data = name_wav.split('.')[-1]

        path_label = os.path.join(self.root,id_data+".npy")

        npy_label = np.load(path_label)
        raw,_ = librosa.load(path_wav,sr=self.fs)

        ## Sampling 

        ## Data
        spec = librosa.stft(raw,window='hann',n_fft=n_fft,hop_length=n_shift, win_length=None,center=True,dtype=np.cdouble)
        mel = np.matmul(mel_basis,np.abs(spec))
        pt = torch.from_numpy(mel)
        pt = pt.float()
        
        # Label
        raw_label = np.zeros(np.shape(mel)[1])
        label = np.zeros(np.shape(raw_label))
        
        for t_label in npy_label :
            idx_start = int(t_label[0]/time_shift)
            idx_end = int(t_label[1]/time_shift)
            #print(str(idx_start) + ' | ' + str(idx_end))
            raw_label[idx_start:idx_end]=1
        
        # if single shift in frame is activate that block is active
        for i in range(len(raw_label)-3):
            if raw_label[i] or raw_label[i+1] or raw_label[i+2] or raw_label[i+3] :
                label[i]=1
        label = torch.from_numpy(label)
        label = label.float()

        return data, label
        #return mel,label

    def __len__(self):
        return len(self.list_path)