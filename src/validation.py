import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics

from utils.hparams import HParam

from tqdm import tqdm

from VAD_dataset import VAD_dataset
from models.RNN_simple import RNN_simple
from models.GPV import GPV
from models.MISO import MISO_1
from models.MISO64 import MISO64
from models.MISO32v2 import MISO32v2
from models.DGD import DGD

## arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True,
                    help="yaml for configuration")
parser.add_argument('--version_name', '-v', type=str, required=True,
                    help="version of current training")
parser.add_argument('--chkpt',type=str,required=False,default=None)
parser.add_argument('--device','-d',type=str,required=False,default='cuda:0')
args = parser.parse_args()

hp = HParam(args.config)
print("NOTE::Loading configuration : "+args.config)


## params
device = args.device
torch.cuda.set_device(device)
print(device)

batch_size = 1
num_workers = hp.train.num_workers

best_loss = 1e6


## data

dataset_test  = VAD_dataset(hp, is_train=False)

loader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=1,shuffle=False,num_workers=num_workers)

print('len test loader : '+str(len(loader_test)))

## model
model = None

channel_in = 1
if 'd' in hp.model.input : 
    channel_in +=1
if 'dd' in hp.model.input : 
    channel_in +=1



if hp.model.type == "GPV":
    model = GPV(hp,channel_in=channel_in,inputdim=hp.model.n_mels,outputdim=1).to(device)
if hp.model.type == "DGD":
    model = DGD(channel_in=channel_in,dim_input=hp.model.n_mels,dim_output=1).to(device)
elif hp.model.type =="MISO":
    num_bottleneck = 5
    en_bottleneck_channels = [1,24,32,64,128,384,64] # 16: 2*Ch 
    Ch = 1  # number of mic
    norm_type = 'IN'  #Instance Norm
    model = MISO_1(num_bottleneck,en_bottleneck_channels,Ch,norm_type).to(device)
elif hp.model.type =="MISO64":
    num_bottleneck = 6
    en_bottleneck_channels = [1,24,32,64,128,256,384] # 16: 2*Ch 
    Ch = 1  # number of mic
    norm_type = 'IN'  #Instance Norm
    model = MISO64(num_bottleneck,en_bottleneck_channels,Ch,norm_type,rate_dropout=hp.model.dropout).to(device)
elif hp.model.type == 'MISO32v2':
    num_bottleneck = 5
    en_bottleneck_channels = [1,24,32,64,128,384,64] # 16: 2*Ch 
    Ch = 1  # number of mic
    norm_type = 'IN'  #Instance Norm
    model = MISO32v2(num_bottleneck,en_bottleneck_channels,Ch,norm_type).to(device)

if model is None :
    raise Exception('No Model specified.')

#model = RNN_simple(hp).to(device)

if not args.chkpt == None : 
   print('NOTE::Loading pre-trained model : '+ args.chkpt)
   model.load_state_dict(torch.load(args.chkpt, map_location=device))
else :
   print('ERROR::no path to model chkpt : '+ args.chkpt)

model.eval()
with torch.no_grad():
    val_loss = 0.0
    f1_score = 0
    #list_threshold = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #list_threshold = [ 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8,0.9]
    list_threshold = [ 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7, 0.8, 0.9]
    list_f1 = np.zeros(len(list_threshold))
    list_acc= np.zeros(len(list_threshold))
    list_tpr = np.zeros(len(list_threshold))

    for j, (data,label) in tqdm(enumerate(loader_test)):
        data = data.to(device)
        label = label.to(device)

        output = model(data)

        for i in range(len(list_threshold)) : 
            label_output = (output[:,0,:] > list_threshold[i]).float()

            # to avoid zero_division error
            label_output[:,0]=1
            label_output[:,1]=0

            pred = torch.squeeze(label_output).cpu().detach().numpy()
            true = torch.squeeze(label[:,0,:]).cpu().detach().numpy()

            list_f1[i] +=sklearn.metrics.f1_score(pred,true)

            list_acc[i] +=sklearn.metrics.accuracy_score(pred,true)
            tp = np.sum((pred==1) & (true==1))
            fn = np.sum((pred == 0) & (true==1))

            # in case of there is no speech in the segment
            fn = np.max([fn,1])
            tp = np.max([tp,1])

            list_tpr[i] += tp/(tp+fn)

    with open('/home/nas/user/kbh/VADK/'+args.version_name+'.csv','w') as f :
        f.write('threshold,f1_score,accuracy,TPR\n')
        for i in range(len(list_threshold)) : 
            list_f1[i] = list_f1[i]/len(loader_test)
            list_acc[i] = list_acc[i]/len(loader_test)
            list_tpr[i] = list_tpr[i]/len(loader_test)
            f.write('{:.2f},{:.4f},{:.4f},{:.4f}\n'.format(list_threshold[i],list_f1[i],list_acc[i],list_tpr[i]))