import torch
import torch.nn as nn


class RNN_simple(nn.Module):
    def __init__(self,hp):
        super(RNN_simple,self).__init__()

        self.hp = hp
        self.n_mels = hp.model.n_mels
        self.dr = hp.model.dropout

        self.lstm =     nn.LSTM(
                input_size = self.n_mels,
                hidden_size = 128,
                num_layers = 2,
                bias = True,
                batch_first = True,
                dropout = self.dr,
                proj_size = 32
            )
        self.activation = nn.ReLU()

        self.layer_2 = nn.Sequential(
            nn.Linear(
                in_features=32,
                out_features=1,
            ),
            nn.ReLU()
        )
        self.out = nn.Sigmoid()

    def forward(self,x):

        # [n_batch,n_mels,n_frame]
        # => [n_batch,n_frame,n_mels]
        #print('1 : ' + str(x.shape))
        x = x.permute(0,2,1)
        #print('2 : ' + str(x.shape))
        x = self.lstm(x)
        x = x[0]
        #print('3 : ' + str(x.shape))
        x = self.activation(x)
        #print('5 : ' + str(x.shape))
        x = self.layer_2(x)
        #print('6 : ' + str(x.shape))
        x = self.out(x)
        #print('7 : ' + str(x.shape))
        x = x[:,:,0]
        return x