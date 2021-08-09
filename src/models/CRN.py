import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_2018(nn.Module):
    def __init__(self,hp):
        super.__init__()

        self.hp = hp
        self.n_mels = hp.model.n_mels
        self.n_frame = hp.model.n_frame

        self.activation

        self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels =1,
                    out_channels =3,
                    kernel_size=(5,5),
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=True
                )
                nn.BatchNorm2d(),
                self.activation,
        )

