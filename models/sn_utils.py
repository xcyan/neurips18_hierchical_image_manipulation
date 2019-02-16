import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import os

def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v, p=2) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
        Apply power iteration for weight parameter
    """
    W = W.view(W.size(0), -1)
    size = W.size() # n x m
    
    _u = u
    for _ in range(Ip):
       _v =  _l2normalize(torch.mm(_u, W)) # 1 x m
       _u = _l2normalize(torch.mm(W, _v.t())) # n x 1
       _u = _u.view(1, -1)

    sigma = _u.mm(W).mm(_v.t())
    return sigma, _u


class SNLinear(nn.Linear):
    # static parameter of iteration time
    # default is 1
    Ip = 1
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        u = Parameter(torch.FloatTensor(1, self.weight.size(0)).normal_(),
                requires_grad=False) # 1 x n
        self.register_parameter("u", u)

    @property
    def W_bar(self):
        sigma, _u = max_singular_value(self.weight, u=self.u, Ip=self.Ip)
        if self.training:
            self.u = Parameter(_u.data, requires_grad=False)
        return self.weight / sigma

    def forward(self, input):
        
        return F.linear(input, self.W_bar, self.bias)

class SNConv2d(nn.Conv2d):
    # static parameter of iteration time
    # default is 1
    Ip = 1
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        u = Parameter(torch.FloatTensor(1, self.weight.size(0)).normal_(),
                requires_grad=False) # 1 x n
        self.register_parameter("u", u)

    @property
    def W_bar(self):
        sigma, _u = max_singular_value(self.weight, u=self.u, Ip=self.Ip)
        if self.training:
            self.u = Parameter(_u.data, requires_grad=False)
        return self.weight / sigma

    def forward(self, input):

        return F.conv2d(input, self.W_bar, self.bias, self.stride, self.padding, 
                self.dilation, self.groups)



