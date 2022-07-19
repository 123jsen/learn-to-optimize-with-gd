import torch
import torch.nn as nn

'''Simple linear regression model'''
class linear_model(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linear_model, self).__init__()
        self.w = nn.Linear(inputSize, outputSize)
        self.b = nn.Parameter(torch.ones(outputSize))

    def forward(self, x):
        out = self.w * x + self.b
        return out