import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dropout = 0.3
        self.input = nn.Linear(input_size, 2048)
        self.h1 = nn.Linear(2048, 2048)
        self.d1 = nn.Dropout(p=self.dropout)
        self.h2 = nn.Linear(2048, 2048)
        self.d2 = nn.Dropout(p=self.dropout)
        self.h3 = nn.Linear(2048, 2048)
        self.d3 = nn.Dropout(p=self.dropout)
        self.output = nn.Linear(2048, 1)

    # NEED TO ADD BATCH NORMALIZATION (either before or after activation function (google said after?? - need to research)) 
    def forward(self, x):
        x = self.input(x)
        x = F.relu(self.h1(x))
        x = self.d1(x)
        x = F.relu(self.h2(x))
        x = self.d2(x)
        x = F.relu(self.h3(x))
        x = F.tanh(self.output(x))

        return x