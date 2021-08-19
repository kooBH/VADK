from VADK.src.model.main_VAD_copy import CNN_BiLSTM
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter
batch_size = 10

## Load data
path_test  ="/home/data/kbh/VADK/test/"

data_test=[]

for data in os.listdir(path_test):
    db = torch.load(os.path.join(path_test,data))
    img, label = db['mel'], db['label']
    data_test.append((img, label))

test_loader = torch.utils.data.DataLoader(data_test, batch_size)

model = CNN_BiLSTM().to(device)

model.load_state_dict(torch.load('model1_cnn.pth'))


## Test ##
with torch.no_grad():
  correct = 0

  for i, (image, label) in enumerate(test_loader):
    image = np.expand_dims(image, axis = 1)
    image = np.expand_dims(image, axis = 1)
    image = torch.FloatTensor(image)

    image = image.to(device)
    label = label.to(device)

    # Forward
    output = model(image)

    _, pred = torch.max(output.data, 1)
    correct += (pred == label).sum().item()

  print('Test Accuracy of VAD model on the {} test images: {}%'.format(len(data_test), 100 * correct / len(data_test)))