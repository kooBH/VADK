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


parser = argparse.ArgumentParser()
parser.add_argument('-i','--dir_input',type=str,required=True)
parser.add_argument('-o','--dir_output',type=str,required=True)
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-d','--device',type=str,default='cuda:0')
parser.add_argument('-n','--num_process',type=int,default=8)
parser.add_argument('-p','--chkpt',type=str,required=True)
args = parser.parse_args()

hp = HParam(args.config)
print('NOTE::Loading configuration :: ' + args.config)

list_data = glob.glob(os.path.join(args.dir_input,'*.wav'))

# Params
device = args.device
torch.cuda.set_device(device)

n_mels = hp.model.n_mels
sr = hp.audio.samplerate
n_fft = hp.audio.frame
n_shift = hp.audio.shift

num_epochs = 1
batch_size = 1

model = None
if hp.model.type == "GPV":
    model = GPV(hp,inputdim=hp.model.n_mels).to(device)
if model is None :
    raise Exception('No Model specified.')

mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

# Dirs
os.makedirs(args.dir_output,exist_ok=True)
# Model 
model.load_state_dict(torch.load(args.chkpt,map_location=device))
model.share_memory()
model.eval()

if not args.chkpt == None : 
   print('NOTE::Loading pre-trained model : '+ args.chkpt)
   model.load_state_dict(torch.load(args.chkpt, map_location=device))


threshold = 0.5

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
        pt = pt.cuda()

        with torch.no_grad():
            output = model(pt)
        output = torch.squeeze(output)
        for idx in range(len(output)):
            if output[idx] < threshold :
                raw[(idx-1)*128:idx*128]=0
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







