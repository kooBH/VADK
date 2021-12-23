import torch
import torch.nn as nn
import pdb


# CNN-BiLSTM
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=0),
            nn.BatchNorm2d(32),  # batchnorm2d(#features)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding = 0))
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding = 0))
            
        self.flatten = nn.Flatten(1,-1)

        self.layer3 = nn.Sequential(
            nn.Linear(32* 6 * 6, 64),
            nn.ReLU())
            # nn.Dropout(0.5))

        self.layer4 = nn.LSTM(64, 32, batch_first = True, bidirectional = True, dropout = 0.5)
            
        self.linear = nn.Linear(31*64 , 31*32)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x = x.unsqueeze(1) #make channel dimension (one channel)

        batch_size, timesteps, C, H, W = x.size()   # [B, T, 1, 32, 32]

        x = x.view(batch_size * timesteps, C, H, W) # [B*T, 1, 32, 32]
        x = self.layer1(x)                          # [B*T, 32, 14, 14]
        x = self.layer2(x)                          # [B*T, 32, 6, 6]
        x = self.flatten(x)                         # [B*T, 32*6*6]
        x = self.layer3(x)                          # [B*T, 64]

        #pdb.set_trace()
        x = x.reshape(batch_size,-1,64)              #[B, T, 64] 64: out-channel of layer3
        output, _ = self.layer4(x)                  # output.shape = [B, T = 1, 64]
        
        output = torch.reshape(output,(output.shape[0],output.shape[1]*output.shape[2]))
        # [n_batch,n_time,n_frame] -> [n_batch,n_time*n_frame]
        x = self.linear(output)
        x = self.softmax(x)
        return x.squeeze(1) # [n_batch,n_timestamp,n_frame]  , [10,31*2 ??]