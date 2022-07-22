import torch
import torch.nn as nn

'''Simple learnable model by variable gradient descent step size'''
class gd_l2o_weight(nn.Module):
    def __init__(self, start_lr):
        super(gd_l2o_weight, self).__init__()
        self.w = nn.Parameter(torch.tensor(start_lr))

    def forward(self, x):
        out = self.w * x
        return out

'''l2o model based on lstm, refer to Androchowicz et al.'''
class lstm_l2o_optimizer(nn.Module):
    def __init__(self):
        super(lstm_l2o_optimizer, self).__init__()
        self.LSTM = nn.LSTM(1, 24, num_layers=2)    # gradient is the sole input
        self.linear = nn.Linear(24, 1)
    
    def forward(self, x, h_in):
        out, h_out = self.LSTM(x, h_in)
        return self.linear(out), h_out