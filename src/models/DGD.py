## Based on https://github.com/RicherMans/GPV 
## Modifed inspired by https://ieeexplore.ieee.org/abstract/document/8552099

import torch
import numpy as np
import torch.nn as nn

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        #print(x.shape)
        return self.block(x)

class DeBlock2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1,padding=1,output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.ConvTranspose2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride,
                      output_padding=output_padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class DGD(nn.Module):
    def  __init__(self,channel_in=1,dim_input = 40, dim_output=1):
        super().__init__()

        self.channel_in = channel_in
        self.dim_input = dim_input
        self.dim_output = dim_output

        self.features = nn.Sequential(
            Block2D(self.channel_in, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            #nn.LPPool2d(4, (1, 4)),
            nn.LPPool2d(4, (1, 2)),
            nn.Dropout(0.3),
        )

        self.up = nn.Sequential(
            DeBlock2D(256, 256,stride=(1,2),output_padding=(0,1)),
            DeBlock2D(256, 128,stride=(1,1),output_padding=0),
            DeBlock2D(128, 128,stride=(1,2),output_padding=(0,1)),
            DeBlock2D(128, 32,stride=(1,1),output_padding=0),
            DeBlock2D(32, 1,stride=(1,1),output_padding=0),
        )

        self.acti = nn.Sigmoid()

        with torch.no_grad():
            # [B,C,T,F]
            rnn_input_dim = self.features(torch.randn(1, channel_in, 500,
                                                      dim_input)).shape
            # C * F 
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
    def forward(self, x):
       # print("IN   : "+str(x.shape))
        x = x.permute(0,1,3,2)
        batch, ch, time, dim = x.shape
        x = self.features(x) # B,128,T',1
        #print("ENC : "+str(x.shape))
        x = x.permute(0,2,1,3)
        x = x.reshape((x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))

        x, _ = self.gru(x) # => [B,]
        #print("RNN : "+str(x.shape))

        x = torch.unsqueeze(x,1)
        x = x.permute(0,3,1,2)

        x = self.acti(x)
        x = self.up(x)
        x = self.acti(x)

        x = torch.squeeze(x,2)
        #print("DEC : "+str(x.shape))

        return x