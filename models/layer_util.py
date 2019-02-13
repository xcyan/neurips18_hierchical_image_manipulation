"""Layer utils file."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np

def weights_init(m, conv_sigma=0.02, bnorm_sigma=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, conv_sigma)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, bnorm_sigma)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class LayerNorm(nn.Module):
    """Implementation of layer normalization"""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        mean = x.mean(1).view(-1, 1)
        std = x.std(1).view(-1, 1)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            y = self.gamma * y + self.beta

        return y


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class SumPooling(nn.Module):
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, input_var):
        size = input_var.size()
        output = torch.sum(input_var.view(size[0], size[1], -1),
                dim=2)
        return output

class Identity(nn.Module):
    def __init__(self, *values):
        super(Identity, self).__init__()

    def forward(self, input_var):
        return input_var

class Piling(nn.Module):
    """
        Spatial Piling the 2-dim tensor: input_var to target size
    """
    def __init__(self, target_size):
        super(Piling, self).__init__()
        self.target_size = target_size

    def forward(self, input_var):
        input_size = input_var.size()
        input_var = input_var.unsqueeze(dim=2).unsqueeze(dim=3)
        output = input_var.expand(input_size[0], input_size[1],
                self.target_size[0], self.target_size[1])
        return output

class ResLinear(nn.Module):
    """Residual block with fully-connected connections"""
    def __init__(self, c_num):
        super(ResLinear, self).__init__()
        self.deep_connect = nn.Sequential(
                    nn.Linear(c_num, c_num / 2),
                    nn.ReLU(),
                    nn.Linear(c_num / 2, c_num / 2),
                    nn.ReLU(),
                    nn.Linear(c_num / 2, c_num),
                    nn.ReLU(),
                )

        self.shortcut_connect = nn.Sequential(
                nn.Linear(c_num, c_num),
                nn.ReLU()
                )

        self.layer_norm = LayerNorm(c_num)

    def forward(self, input_var):
        output = self.shortcut_connect(input_var) + self.deep_connect(input_var)

        output = self.layer_norm(output)

        return output

class ConvResnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_layers=1, stride=1,
            kernel_size=3, norm_fn=None, activation_fn=None):
        super(ConvResnetBlock, self).__init__()
        assert norm_fn
        assert activation_fn
        ###############################
        ## Build Shortcut connection ##
        ###############################
        if in_planes == out_planes and stride == 1:
            self.shortcut = None
        else:
            self.shortcut = [
                nn.Conv2d(in_planes, out_planes, kernel_size=1,
                    stride=stride),
                norm_fn(out_planes)]
            self.shortcut = nn.Sequential(*self.shortcut)
        ###########################
        ## Build Deep connection ##
        ###########################
        self.deep = []
        padding = int((kernel_size-1)/2)
        self.deep += [
            activation_fn,
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding),
            norm_fn(out_planes)]
        for i in xrange(1, num_layers):
            self.deep += [
                activation_fn,
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                    stride=1, padding=padding),
                norm_fn(out_planes)]
        self.deep = nn.Sequential(*self.deep)

    def forward(self, x):
        residual = x
        out = self.deep(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out = out + residual
        return out

class DeconvResnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_layers=1, stride=1, kernel_size=3,
                 norm_fn=None, activation_fn=None):
        super(DeconvResnetBlock, self).__init__()
        assert norm_fn
        assert activation_fn
        ###############################
        ## Build shortcut connection ##
        ###############################
        if out_planes == in_planes and stride == 1:
            self.shortcut = None
        else:
            self.shortcut = []
            if in_planes != out_planes:
                self.shortcut += [
                    nn.Conv2d(in_planes, out_planes, kernel_size=1),
                    norm_fn(out_planes)]
            if stride > 1:
                self.shortcut += [nn.Upsample(scale_factor=stride, mode='bilinear')]
            self.shortcut = nn.Sequential(*self.shortcut)
        ###########################
        ## Build Deep connection ##
        ###########################
        self.deep = []
        if kernel_size % 2 == 1:
            self.build_conv2d_block(in_planes, out_planes, num_layers, stride,
                                    kernel_size, norm_fn, activation_fn)
        else:
            self.build_tconv2d_block(in_planes, out_planes, num_layers, stride,
                                     kernel_size, norm_fn, activation_fn)
        self.deep = nn.Sequential(*self.deep)

    def build_conv2d_block(self, in_planes, out_planes, num_layers, stride,
                           kernel_size, norm_fn, activation_fn):
        if stride > 1:
            self.deep.append(nn.Upsample(scale_factor=stride, mode='bilinear'))
        padding = int((kernel_size-1)/2)
        self.deep += [
            activation_fn,
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=1, padding=padding),
            norm_fn(out_planes)]
        for i in xrange(num_layers-1):
            self.deep += [
                activation_fn,
                nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size,
                    stride=1, padding=padding),
                norm_fn(out_planes)]

    def build_tconv2d_block(self, in_planes, out_planes, num_layers, stride,
                            kernel_size, norm_fn, activation_fn):
        if stride == 1:
            padding = int(kernel_size/2)
            output_padding = stride
            self.deep += [
                activation_fn,
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, output_padding=output_padding),
                norm_fn(out_planes)]
        else:
            padding = int((kernel_size-1)/2)
            output_padding = stride - 2
            self.deep += [
                activation_fn,
                nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, output_padding=output_padding),
                norm_fn(out_planes)]
        #
        for i in xrange(num_layers-1):
            padding = int(kernel_size/2)
            self.deep += [
                activation_fn,
                nn.ConvTranspose2d(out_planes, out_planes, kernel_size=kernel_size,
                    stride=1, padding=padding, output_padding=1),
                norm_fn(out_planes)]

    def forward(self, x):
        residual = x
        out = self.deep(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        out = out + residual
        return out

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class DilatedResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True, activation_fn=nn.InstanceNorm2d):
        super(DilatedResnetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = activation_fn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = activation_fn(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out

class FeatureFusionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, fusion_type,
                 norm_fn, activation_fn, main_module):
        super(FeatureFusionBlock, self).__init__()
        #assert in_planes == out_planes
        assert fusion_type in ['add', 'concat'] # 'deep']
        self.fusion_type = fusion_type
        self.main_module = main_module
        self.norm_fn = norm_fn
        self.activation_fn = activation_fn
        if fusion_type == 'concat':
            self.fuse_module = self.initialize_concat_layer(in_planes)
        else:
            self.fuse_module = self.initialize_deep_layer(in_planes)

    def initialize_concat_layer(self, in_planes):
        self.nonlinear1 = self.activation_fn
        self.conv1 = nn.Conv2d(in_planes * 2, in_planes,
            kernel_size=1, stride=1, padding=0)
        self.norm1 = self.norm_fn(in_planes)

    #def initialize_deep_layer(self, in_planes, out_planes):
    def initialize_deep_layer(self, in_planes):
        pass

    def forward(self, x, y):
        #out = self.main_module(x)
        if self.fusion_type == 'add':
            out = x + y
        elif self.fusion_type == 'concat':
            out = torch.cat([x, y], 1)
            out = self.nonlinear1(out)
            out = self.conv1(out)
            out = self.norm1(out)
        out = self.main_module(out)
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer,
            activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
                dim, padding_type, norm_layer,
                activation, use_dropout)

    def build_conv_block(self, dim, padding_type,
            norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                    'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                    'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


