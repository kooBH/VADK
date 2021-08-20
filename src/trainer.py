import torch
import torch.nn as nn
import argparse
import os

from tensorboardX import SummaryWriter
from utils.hparams import HParam
from utils.writer import MyWriter

from VAD_dataset import VAD_dataset

from models.RNN_simple import RNN_simple

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

dataset_train = VAD_dataset(hp.data.root, is_train=True)
dataset_test  = VAD_dataset(hp.data.root, is_train=False)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=1,shuffle=False,num_workers=num_workers)

## model
model = None

## TODO
if hp.model == "A":
    #model = A().to(device)
    pass
elif hp.model =="B":
    #model = B().to(device)
    pass

model = RNN_simple(hp).to(device)

if not args.chkpt == None : 
   print('NOTE::Loading pre-trained model : '+ args.chkpt)
   model.load_state_dict(torch.load(args.chkpt, map_location=device))
    
criterion = nn.BCELoss()
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
            steps_per_epoch = len(train_loader)
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

        data = data.to(device)
        label = label.to(device)

        # Forward
        output = model(data)

        loss = criterion(output, label)

        # Backward and optimize
        # scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx+1) % hp.train.summary_interval == 1:
            print("Epoch [{}/{}], Step[{}], Loss:{:.4f}, best{:.4f}".format(epoch+1, num_epochs, idx+1, loss.item(),best_loss))
        break
    
    ## Eval
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for j, (batch_data) in enumerate(loader_test):

            print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(loader_test), loss.item()))
            val_loss +=loss.item()
            break

        val_loss = val_loss/len(loader_test)
        scheduler.step(val_loss)

        writer.log_value(loss,step,'test loss['+hp.loss.type+']'    )


    torch.save(model.state_dict(), "lastmodel.pt")
    if best_loss >  val_loss:
        torch.save(model.state_dict(), "bestmodel.pt")
        best_loss = val_loss

    ## Log

        

