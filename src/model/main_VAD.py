from tqdm import tqdm
import torch
import torch.nn as nn
import pdb
from data_loader import VAD_dataset
from CNN_biLSTM import CNN_BiLSTM
import argparse


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--device','-d',type=str,required=False,default='cuda:0',help="specify cuda device")
args = parser.parse_args()


# parameter
num_epochs = 100
learning_rate = 2*1e-3
batch_size = 256
num_workers = 8
device = args.device

## Load data
#root ="/home/data/kbh/VADK/"
root = "/home/hido/data_tmp/" 

train_dataset = VAD_dataset(root,is_train=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

## TODO test_dataset, test_dataloader

model = CNN_BiLSTM().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#scheduler = torch.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)


## Train ##
total_step = len(train_loader)
loss_list = []

bestloss = 1e06

# for epoch in tqdm(range(num_epochs)):
for epoch in range(num_epochs):
  model.train()

  for idx,(img,label) in enumerate(train_loader):
    ## batch = [n_batch, n_timestamp, n_mels, n_frame]
  
    ## [n_batch,n_timestamp,n_mels,n_frame]
    ##       -> [n_batch,n_timestamp,n_channel,n_mels,n_frame]
    img = torch.unsqueeze(img,dim=2)
    
    img = img.to(device).float()
    label = label.to(device).float()

    # Forward
    output = model(img)
    
    loss = criterion(output, label)

    # Backward and optimize
    # scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (idx+1) % 20 == 1:
        print("Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}, best{:.4f}".format(epoch+1, num_epochs, idx+1, total_step, loss.item(),bestloss))

    if bestloss >  loss.item():
        torch.save(model.state_dict(), "model.pth")
        bestloss = loss.item()
