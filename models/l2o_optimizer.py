import torch
import torch.nn as nn

'''Simple learnable model by variable gradient descent step size'''
class l2o_optimizer(nn.Module):
    def __init__(self, start_lr):
        super(l2o_optimizer, self).__init__()
        self.w = nn.Parameter(torch.tensor(start_lr))

    def forward(self, x):
        out = self.w * x
        return out