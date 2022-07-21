import torch
import torch.nn as nn
from torch.nn import functional as F

'''Simple classification network for MNIST dataset'''
class class_net(nn.Module):
    def __init__(self, inputNum, outputNum):
        super(class_net, self).__init__()
        self.layer1 = nn.Linear(inputNum, 32)
        self.layer2 = nn.Linear(32, outputNum)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        return x