# import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pdb


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter
num_epochs = 10000
learning_rate = 0.05
batch_size = 2

dataset=[]

# Load data
data_path = '/home/hido/space/VADK/src/model/sample_32by32_mel/'
for data in os.listdir(data_path):
    db = torch.load(os.path.join(data_path,data))
    img, label = db['mel'], db['label']
    dataset.append((img, label))

train_loader = torch.utils.data.DataLoader(dataset, batch_size)


# CNN-BiLSTM
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=0),   # B * 1 * 32 * 32  => B * 32 * 28 * 28
            nn.BatchNorm2d(32),  # batchnorm2d(#features)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding = 0))     # B * 32 * 14 * 14

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=0),  # B * 32 * 12 * 12
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding = 0))     # B * 32 * 6 * 6
            
        self.flatten = nn.Flatten(1,-1)                 # B * 32 * (6*6)

        self.layer3 = nn.Sequential(
            nn.Linear(32* 6 * 6, 64),
            nn.ReLU())
            # nn.Dropout(0.5))

        self.layer4 = nn.Sequential( # 최종 데이터의 shape이 (size, timestamp, feature)
            nn.LSTM(64, 32, batch_first = True, bidirectional = True))
            # nn.Dropout(0.5))

        self.linear = nn.Linear(64 , 32) # 32개의 label과 비교하기 때문
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.unsqueeze(1) #make channel dimension (one channel)
        batch_size, timesteps, C, H, W = x.size()   # [B, 1, 1, 32, 32]

        x = x.view(batch_size * timesteps, C, H, W) # [B, 1, 32, 32]
        x = self.layer1(x)                          # [B, 32, 14, 14]
        x = self.layer2(x)                          # [B, 32, 6, 6]
        x = self.flatten(x)                         # [B, 32*6*6]
        x = self.layer3(x)                          # [B, 64]

        x = x.reshape(batch_size,1,-1)              #[B, T = 1, 64] 64: out-channel of layer3
        output, _ = self.layer4(x)                  # output.shape = [B, T = 1, 64]

        x = self.linear(output[:, -1, :])
        x = self.softmax(x)
        return x.squeeze(1)


# sequence_length = 15 (뜬금 없이??????????)
def time_distribute(data, sequence_length: int, stride: int = None, z_pad: bool = True) -> np.ndarray:
    if stride is None:
        stride = sequence_length
    if stride > sequence_length:
        print('WARNING: Stride longer than sequence length, causing missed samples. This is not recommended.')
    td_data = []
    for n in range(0, len(data)-sequence_length+1, stride):
        td_data.append(data[n:n+sequence_length])
    if z_pad:
        if len(td_data)*stride+sequence_length != len(data)+stride:
            z_needed = len(td_data)*stride+sequence_length - len(data)
            z_padded = np.zeros(td_data[0].shape)
            for i in range(sequence_length-z_needed):
                z_padded[i] = data[-(sequence_length-z_needed)+i]
            td_data.append(z_padded)
    return np.array(td_data)


model = CNN_BiLSTM().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#scheduler = torch.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)

# train
# start = time.time()  # train 얼마나 걸렸는지 보기위해 초단위 저장
total_step = len(train_loader)
loss_list = []

# for epoch in tqdm(range(num_epochs)):
for epoch in range(num_epochs):
  for i, (image, label) in enumerate(train_loader):
    image = np.expand_dims(image, axis = 1)
    image = np.expand_dims(image, axis = 1)
    image = torch.FloatTensor(image)

    image = image.to(device)
    label = label.to(device)

    # Forward
    output = model(image)
    loss = criterion(output, label)

    # Backward and optimize
    # scheduler.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())

    if (i+1) % 1 == 0:
      print("Epoch [{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch+1, num_epochs, i+1, total_step, loss.item()))
