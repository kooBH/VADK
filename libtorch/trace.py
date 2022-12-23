import torch
import torch.onnx
import torch.nn as nn
import argparse

import sys,os 
sys.path.append(os.path.abspath("../src"))

from utils.hparams import HParam
from models.GPV import GPV


parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-p','--chkpt',type=str,required=True)
args = parser.parse_args()

# import test_model
# load pre-trained model ------------------------------------
hp = HParam(args.config)
model = GPV(hp,channel_in=3,inputdim=hp.model.n_mels,outputdim=1)
model.load_state_dict(torch.load(args.chkpt))
print(args.chkpt)
model.eval()

# data for tracing ---------------------------------------

# [B, C, F, T ]
input = torch.rand(1,3,hp.model.n_mels,50,requires_grad=True)
print(type(input))
print(input.shape)

# inference stage -------------------------------------------
output = model(input)
print(output.shape)

traced_model = torch.jit.trace(model, input)
traced_model.save('GPV.pt')


# ONNX

print("===== ONX =====")

"""
https://yunmorning.tistory.com/17 -> old method? 

https://docs.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-convert-model
"""

# tracing metohd
torch.onnx.export(
        model,         # model being run 
        input,       # model input (or a tuple for multiple inputs) 
        "GPV.onnx",       # where to save the model  
        opset_version=11,
        #vervose=False,
        input_names = ['input'], 
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size',3: 'n_unit'},    # variable length axes
                                'output' : {0 : 'batch_size',3: 'n_unit'}},
        export_params=True
        )

print("---- EXPORTED ---- ")
