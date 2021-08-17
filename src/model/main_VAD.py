import torch
import torch.nn as nn
import torch.optim as optim
import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parameter
num_epochs = 1
learning_rate = 0.05
batch_size = 64

# Load data


# Define DataLoader


# CNN-BiLSTM
class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()
        self.layer1 = nn.Sequential( # 32*32*3 Batch 개수로 묶어서 -> B*32*32*3 -> 5 X 5 (stride - ) ->  -> maxpool -> ->  3 X 3 -> # B * 32 * 32 128 => Maxpool 64 Dense B개수의 Bi LSTM 64 + 64 =128
            nn.Conv2d(3, 32, 5, stride=1, padding=2),  # B * 3 * 32 * 32  => B * 32 * 32 * 32
            nn.BatchNorm2d(32),  # batchnorm2d(#features)
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1, padding = 1)) #B * 32 * 32 * 32
        self.layer2 = nn.Sequential( # 14*14*32 -> 14*14*64 -> 7*7*64
            nn.Conv2d(32, 32, 3, stride=1, padding=1), # B 32 32 32
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1, padding = 1)) #B 32 32 32
        self.flatten = nn.Flatten(1, 2)
        self.layer3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU())
        self.layer4 = nn.LSTM(64, 32, batch_first = True, bidirectional = True)
        self.linear = nn.Linear(64, 1)
        self.softmax = nn.Softmax(dim=2)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.layer3(x)
        output, _ = self.layer4(x)
        x = self.linear(x)
        x = self.softmax(output)
        return x


model = CNN_BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#scheduler = torch.lr_scheduler.ExponentialLR(optimizer, gamma= 0.99)
image = torch.rand(batch_size, 3, 32, 32) # 일단 랜덤 값을 넣음
output = model(image)       
print(output)