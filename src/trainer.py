import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics

from tensorboardX import SummaryWriter
from utils.hparams import HParam
from utils.writer import MyWriter

from tqdm import tqdm

from VAD_dataset import VAD_dataset
from models.RNN_simple import RNN_simple

# from models.GPV import GPV
from models.MISO32v2 import MISO32v2
from models.MISO64 import MISO64
#from models.MISO_stft import MISO_stft
from models.GPV import GPV
from models.DGD import DGD

from utils.specaugmentation import spec_augment

def aug(hp,data):
    return data

## arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True,
                    help="yaml for configuration")
parser.add_argument('--version_name', '-v', type=str, required=True,
                    help="version of current training")
parser.add_argument('--chkpt',type=str,required=False,default=None)
parser.add_argument('--step','-s',type=int,required=False,default=0)
parser.add_argument('--device','-d',type=str,required=False,default='cuda:0')
args = parser.parse_args()

hp = HParam(args.config)
print("NOTE::Loading configuration : "+args.config)


## params
device = args.device
torch.cuda.set_device(device)
print(device)

batch_size = hp.train.batch_size
num_epochs = hp.train.epoch
num_workers = hp.train.num_workers

best_loss = 1e6


## path
modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

os.makedirs(modelsave_path,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)

## logger
writer = MyWriter(hp, log_dir)

## data

dataset_train = VAD_dataset(hp, is_train=True)
dataset_test  = VAD_dataset(hp, is_train=False)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=1,shuffle=False,num_workers=num_workers)

print('len train loader : '+str(len(loader_train)))
print('len test loader : '+str(len(loader_test)))

## model
model = None
output_dim = 1

channel_in = 1
if 'd' in hp.model.input : 
    channel_in +=1
if 'dd' in hp.model.input : 
    channel_in +=1


if hp.model.type == "GPV":
    model = GPV(hp,channel_in=channel_in,inputdim=hp.model.n_mels,outputdim=output_dim).to(device)
elif hp.model.type == "DGD":
    model = DGD(channel_in=channel_in,dim_input=hp.model.n_mels,dim_output=output_dim).to(device)
elif hp.model.type =="MISO32":
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
    model = MISO64(num_bottleneck,en_bottleneck_channels,Ch,norm_type).to(device)
elif hp.model.type =="MISO_stft":
    pass
    num_bottleneck = 32
    en_bottleneck_channels = [1,24,32,64,128,256,384] # 16: 2*Ch 
    Ch = 1  # number of mic
    norm_type = 'IN'  #Instance Norm
    model = MISO_stft(num_bottleneck,en_bottleneck_channels,Ch,norm_type).to(device)
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

if hp.loss.type == 'BCELoss' :
    criterion = nn.BCELoss()
elif hp.loss.type == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(hp.loss.BCEWithLogitsLoss.pos_weight))
else:
    raise Exception('No Such Loss ' + str(hp.loss.type))


optimizer = torch.optim.Adam(model.parameters(), lr=hp.optim.Adam)

if hp.scheduler.type == 'Plateau': 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        mode=hp.scheduler.Plateau.mode,
        factor=hp.scheduler.Plateau.factor,
        patience=hp.scheduler.Plateau.patience,
        min_lr=hp.scheduler.Plateau.min_lr)
elif hp.scheduler.type == 'oneCycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
            max_lr = hp.scheduler.oneCycle.max_lr,
            epochs=hp.train.epoch,
            steps_per_epoch = len(loader_train)
            )
else :
    raise TypeError("Unsupported scheduler type")


step = args.step
# to detect NAN
torch.autograd.set_detect_anomaly(True)

print('NOTE::Training starts.')

step = 0
for epoch in range(num_epochs):
    model.train()
    loss_train = 0

    ## Train    
    for idx,(data,label) in enumerate(loader_train):
        step += 1
        ## img = [n_batch, n_mels, n_frame]

        data=aug(hp,data)

        data = data.to(device)
        label = label.to(device)

        #print(data.shape)
        # Forward
        output = model(data)
        loss = criterion(output.float(), label.float())

        # Backward and optimize
        # scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % hp.train.summary_interval == 1:
            print("TRAIN::{}: Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}".format(args.version_name,epoch+1, num_epochs, idx+1,len(loader_train), loss.item()))

    ## Eval
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        f1_score = 0
        #list_threshold = [0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        list_threshold = [ 0.1, 0.2, 0.3,  0.4, 0.5, 0.6, 0.7, 0.8,0.9]
        list_f1 = np.zeros(len(list_threshold))
        list_acc= np.zeros(len(list_threshold))
        list_tpr = np.zeros(len(list_threshold))

        for j, (data,label) in tqdm(enumerate(loader_test)):
            #data=aug(hp,data)

            data = data.to(device)
            label = label.to(device)
            output = model(data)


            loss = criterion(output.float(), label.float())
            val_loss +=loss.item()

            for i in range(len(list_threshold)) : 
                label_output = (output[:,0,:]  > list_threshold[i]).float()

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
            #print(output[0,0:10])
            #print(label_output[0:10])
            #print(label[0][0:10])

        val_loss = val_loss/len(loader_test)
        if hp.scheduler.type == 'Plateau' : 
            scheduler.step(val_loss)
        else :
            scheduler.step()




        print('TEST::{}:Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(args.version_name,epoch+1, num_epochs, j+1, len(loader_test), val_loss))

        # logging

        writer.log_value(val_loss,step,'test loss['+hp.loss.type+']'    )
        #writer.log_value(f1_score,step,'test f1_score'    )

        print('--threshold---f1_score---accuracy---TPR--')
        for i in range(len(list_threshold)) : 
           list_f1[i] = list_f1[i]/len(loader_test)
           list_acc[i] = list_acc[i]/len(loader_test)
           list_tpr[i] = list_tpr[i]/len(loader_test)
           #print( 'thr : '+str(list_threshold[i])+' | f1 : ' + str(list_f1[i]) +' | acc : ' + str(list_acc[i]))
           #print('thr : {:.2f} | f1 : {:.4f} | acc : {:.4f}'.format(list_threshold[i],list_f1[i],list_acc[i]))
           print('|   {:.2f}    |  {:.4f} |  {:.4f}  | {:.4f} | '.format(list_threshold[i],list_f1[i],list_acc[i],list_tpr[i]))


    torch.save(model.state_dict(), modelsave_path +"/lastmodel.pt")
    if best_loss >  val_loss:
        torch.save(model.state_dict(), modelsave_path + "/bestmodel.pt")
        best_loss = val_loss

    ## Log