import os, glob
import torch
import numpy as np

class dataset(torch.utils.data.Dataset):
    def __init__(self, hp, inference=None) : 
        self.hp = hp
        self.root = hp.data.root


    def __gettime__(self,index):

    def __len__(self):