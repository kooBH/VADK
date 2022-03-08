import torch
import argparse
import os,glob
import librosa
import soundfile as sf
import numpy as np
from utils.hparams import HParam
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
from models.GPV import GPV
from models.DGD import DGD


parser = argparse.ArgumentParser()
parser.add_argument('-i','--dir_input',type=str,required=True)
parser.add_argument('-o','--dir_output',type=str,required=True)
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-d','--device',type=str,default='cuda:0')
parser.add_argument('-n','--num_process',type=int,default=8)
parser.add_argument('-p','--chkpt',type=str,required=True)
parser.add_argument('-t','--threshold',type=float,default=0.5)
parser.add_argument('-u','--unit',type=int,default=-1)

args = parser.parse_args()

hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

print('inference : {}'.format(args.dir_input))
list_data = glob.glob(os.path.join(args.dir_input,'*.wav'))
print('length : ' + str(len(list_data)))

if len(list_data) == 0:
    raise Exception('Zero targets....')

# Params
device = args.device
torch.cuda.set_device(device)
unit = args.unit

n_mels = hp.model.n_mels
sr = hp.audio.samplerate
n_fft = hp.audio.frame
n_shift = hp.audio.shift

num_epochs = 1
batch_size = 1

channel_in = 1
if 'd' in hp.model.input : 
    channel_in +=1
if 'dd' in hp.model.input : 
    channel_in +=1

model = None
if hp.model.type == "GPV":
    model = GPV(hp,channel_in=channel_in,inputdim=hp.model.n_mels,outputdim=hp.model.label).to(device)
elif hp.model.type == "DGD":
    model = DGD(channel_in=channel_in,dim_input=hp.model.n_mels,dim_output=hp.model.label).to(device)
if model is None :
    raise Exception('No Model specified.')

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# Model 
model.load_state_dict(torch.load(args.chkpt,map_location=device))
model.share_memory()
model.eval()

if not args.chkpt == None : 
   print('NOTE::Loading pre-trained model : '+ args.chkpt)
   model.load_state_dict(torch.load(args.chkpt, map_location=device))

threshold = args.threshold

def inference(batch):
    for idx in batch :
        path_target = list_data[idx]
        name_target = path_target.split('/')[-1]

        raw, _ = librosa.load(path_target,sr=sr)
        spec = librosa.stft(raw,window='hann',n_fft=n_fft,hop_length=n_shift, win_length=None,center=True,dtype=np.cdouble)
        mel = np.matmul(mel_basis,np.abs(spec))
        pt = torch.from_numpy(mel)
        pt = pt.float()
        pt = torch.unsqueeze(pt,dim=0)

        shape = pt.shape

        if 'd' in hp.model.input :
            d = torch.zeros(shape)
            # C, dim, T
            d[:,:-1,:] = pt[0,1:,:]-pt[0,0:-1,:]
            # channel-wise concat
            pt = torch.cat((pt,d),0)
        
        if 'dd' in hp.model.input :
            dd = torch.zeros(shape)
            dd[:,:-2,:] = pt[0,1:-1,:]-pt[0,0:-2,:]
            # channel-wise concat
            pt = torch.cat((pt,dd),0)

        pt = torch.unsqueeze(pt,dim=0)
        pt = pt.cuda()

        # pt [ B , C , F, T]

        # offline  
        if unit == -1 : 
            with torch.no_grad():
                output = model(pt)
            output = torch.squeeze(output)
            for idx2 in range(len(output)):
                if output[idx2] < threshold :
                    raw[(idx2-1)*128:idx2*128]=0
        # online 
        else :  
            n_unit = int(np.ceil(pt.shape[3]/unit))
            # process per unit-frame
            with torch.no_grad():
                for i in range(n_unit) : 
                    if i == n_unit-1 :
                        continue
                    else :
                        tmp_pt = pt[:,:,:,i*unit:]

                    output = model(tmp_pt)
                    output = torch.squeeze(output)
                    for idx2 in range(len(output)):
                        if output[idx2] < threshold :
                            raw[(i*unit+idx2-1)*128:(i*unit+idx2)*128]=0
            

        sf.write(os.path.join(args.dir_output,name_target),raw,16000)

if __name__ == '__main__':
    set_start_method('spawn')

    processes = []
    batch_for_each_process = np.array_split(range(len(list_data)),args.num_process)

    os.makedirs(args.dir_output,exist_ok=True)

    for worker in range(args.num_process):
        p = mp.Process(target=inference, args=(batch_for_each_process[worker][:],) )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()







