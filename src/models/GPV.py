# https://github.com/RicherMans/GPV 

import torch
import numpy as np
import torch.nn as nn

def crnn(inputdim=32, outputdim=527):
    model = CRNN(inputdim, outputdim)
    return model


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


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)


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




class GPV(nn.Module):
    def __init__(self, hp,inputdim=32, outputdim=1, **kwargs):
        super().__init__()
        features = nn.ModuleList()

        self.hp = hp

        self.features = nn.Sequential(
            Block2D(1, 32),
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
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
                                               inputdim=256,
                                               outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        x = x.permute(0,2,1)
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision_time = torch.nn.functional.interpolate(
            decision_time.transpose(1, 2),
            time,
            mode='linear',
            align_corners=False).transpose(1, 2)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)

        #print('-----')
        #print(decision.shape)
        #print(decision)
        #print('=====')
        #print(decision_time.shape)
        #print(decision_time)
        #exit()

        #return decision, decision_time
        #print(decision_time[0,0:10,0])
        return decision_time[:,:,0]