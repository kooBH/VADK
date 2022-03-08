import torch
import torch.nn as nn
import onnx
import argparse

import sys,os 
sys.path.append(os.path.abspath("../src"))

from utils.hparams import HParam
from models.GPV import GPV


parser = argparse.ArgumentParser()
parser.add_argument('-c','--config',type=str,required=True)
parser.add_argument('-p','--chkpt',type=str,required=True)
parser.add_argument('-d','--device',type=str,default='cuda:0')
args = parser.parse_args()

device = args.device

# import test_model
# load pre-trained model ------------------------------------
hp = HParam(args.config)
model = GPV(hp,channel_in=3,inputdim=hp.model.n_mels,outputdim=1).to(device)
model.load_state_dict(torch.load(args.chkpt,map_location=device))
model.eval()

# data for tracing ---------------------------------------

# [B, C, F, T ]
input = torch.rand(1,3,hp.model.n_mels,50).to(device)
print(type(input))
print(input.shape)

# inference stage -------------------------------------------
output = model(input)
print(output.shape)
torch.onnx.export(model, input, "GPV.onnx", input_names=["input"], output_names=["output"], export_params=True,opset_version=11)

print("---- EXPORTED ---- ")

onnx_model = onnx.load("GPV.onnx")
# check that the model converted fine
onnx.checker.check_model(onnx_model)

print("Model was successfully converted to ONNX format.")
print("It was saved to", "GPV.onnx")