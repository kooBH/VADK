import torch
import torch.nn as nn

import sys,os 
sys.path.append(os.path.abspath("../src"))

from utils.hparams import HParam
from models.GPV import GPV

# import test_model
# load pre-trained model ------------------------------------
hp = HParam("../config/GPV_10.yaml")
model = GPV(hp,channel_in=3,inputdim=hp.model.n_mels,outputdim=1)
model.load_state_dict(torch.load("/home/nas/user/kbh/VADK/chkpt/GPV_10/bestmodel.pt"))
model.eval()

# data for tracing ---------------------------------------

# [B, C, F, T ]
#input = torch.rand(1,3,hp.model.n_mels,50)
input = torch.ones(1,3,hp.model.n_mels,50)
print(type(input))
print(input.shape)

# inference stage -------------------------------------------
output = model(input)
print(output)

