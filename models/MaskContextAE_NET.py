"""Factory for box-conditioned mask context auto-encoder."""

import torch
import torch.nn as nn
import torch.nn.parallel
import functools
from torch.autograd import Variable
import numpy as np
from layer_util import *

class MaskContextAE_NET(nn.Module):
    def __init__(self, opt):
        super(MaskContextAE_NET, self).__init__()
        self.input_nc = opt.label_nc
        self.output_nc = opt.output_nc
        self.img_size = opt.fineSize
  
        self.num_layers = opt.num_layers
        self.conv_dim = opt.conv_dim
        self.conv_size = opt.conv_size
        self.embed_dim = opt.embed_dim
        self.z_dim = opt.z_dim
  
        self.norm_fn = get_norm_layer(opt.norm_layer)
        self.use_dropout = opt.use_dropout # TODO(xcyan): add dropout
        self.skip_start = opt.skip_start
        self.skip_end = opt.skip_end
        self.use_resnetblock = opt.use_resnetblock
        self.num_resnetblocks = opt.num_resnetblocks
        self.fusion_type = opt.fusion_type
  
        self.first_conv_stride = opt.first_conv_stride
        self.first_conv_size = opt.first_conv_size 
  
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.eval_mode = 0
 
    def initialize(self):
        self.conv_encoder_modules = self.get_conv_encoder()
        self.conv_decoder_modules = self.get_conv_decoder()
        self.latent_encoder = self.get_latent_encoder()
        self.latent_decoder = self.get_latent_decoder()
  
        self.params_dict = self.get_params_dict()
        self.trainable_parameters = self.get_trainable_parameters()
 
    def get_params_dict(self, scope=None):
        params_dict = dict()
        encoder_scope = (scope is None) or (scope == 'E')
        decoder_scope = (scope is None) or (scope == 'G')
        if encoder_scope:
            for i, conv_module in enumerate(self.conv_encoder_modules):
                params_dict['conv_encoder_%d' % i] = conv_module
            params_dict['latent_encoder'] = self.latent_encoder
        if decoder_scope:
            for i, conv_module in enumerate(self.conv_decoder_modules):
                params_dict['conv_decoder_%d' % i] = conv_module
            params_dict['latent_decoder'] = self.latent_decoder
        return params_dict
    
    def get_trainable_parameters(self, params_dict=None):
        if params_dict is None:
            params_dict = self.params_dict
        assert params_dict
        trainable_parameters = []
        for _, v in params_dict.iteritems():
            trainable_parameters += list(v.parameters())
        return trainable_parameters

    def cudafy(self, gpu_id):
        # move your model to gpu
        for _, v in self.params_dict.iteritems():
            v.cuda(gpu_id)

    def get_conv_encoder(self):
        conv_modules = []
        activation_fn = nn.LeakyReLU(0.2, True)
        kernel_size = self.first_conv_size
        if self.first_conv_stride == 1:
            curr_module = [
                nn.ReflectionPad2d((kernel_size-1)/2),
                nn.Conv2d(self.input_nc, self.conv_dim, kernel_size=kernel_size, padding=0),
                self.norm_fn(self.conv_dim)]
        else:
            curr_module = [
                nn.Conv2d(self.input_nc, self.conv_dim, kernel_size=kernel_size,
                          padding=(kernel_size-1)/2, stride=self.first_conv_stride),
                self.norm_fn(self.conv_dim)]
        curr_module = nn.Sequential(*curr_module)
        conv_modules.append(curr_module)
        for i in xrange(self.num_layers):
            if i >= 3:
                input_dim = (2**3) * self.conv_dim
                output_dim = input_dim
            else:
                input_dim = (2**i) * self.conv_dim
                output_dim = 2 * input_dim

            curr_module = ConvResnetBlock(
                input_dim, output_dim, stride=2,
                kernel_size=self.conv_size, num_layers=self.num_resnetblocks,
                norm_fn=self.norm_fn,
                activation_fn=activation_fn)
            conv_modules.append(curr_module)
        return conv_modules

    # TODO(xcyan): refactor the fusion_type
    def get_conv_decoder(self):
        conv_modules = []
        activation_fn = nn.ReLU(True)
        for i in xrange(self.num_layers):
            if self.num_layers - i >= 4:
              input_dim = (2**3) * self.conv_dim
              output_dim = input_dim
            else:
              input_dim = (2**(self.num_layers - i)) * self.conv_dim
              output_dim = int(input_dim / 2)
            #if self.use_resnetblock and input_dim == output_dim:
            curr_module = DeconvResnetBlock(
                input_dim, output_dim, stride=2, kernel_size=self.conv_size,
                num_layers=self.num_resnetblocks,
                norm_fn=self.norm_fn,
                activation_fn=activation_fn)
            use_skip = (self.skip_start <= i) and (i <= self.skip_end) and (self.fusion_type != 'add')
            if not use_skip:
                conv_modules.append(curr_module)
            else:
                fusion_module = FeatureFusionBlock(
                    input_dim, output_dim, fusion_type=self.fusion_type,
                    norm_fn=self.norm_fn, activation_fn=activation_fn,
                    main_module=curr_module)
                conv_modules.append(fusion_module)
        
        kernel_size = self.first_conv_size
        conv_stride = self.first_conv_stride
        padding = (kernel_size - 1) / 2
        if self.first_conv_stride == 1:
            curr_module = [
                activation_fn,
                nn.Conv2d(self.conv_dim, self.output_nc,
                    kernel_size=kernel_size, padding=padding)]
            curr_module = nn.Sequential(*curr_module)
        else:
            output_padding = conv_stride + 2 * padding - kernel_size
            curr_module = [
                activation_fn,
                nn.ConvTranspose2d(
                    self.conv_dim, self.output_nc, kernel_size=kernel_size,
                    stride=conv_stride, padding=padding,
                    output_padding=output_padding)]
            curr_module = nn.Sequential(*curr_module)
        
        use_skip = (self.skip_start <= self.num_layers) and (self.num_layers <= self.skip_end) \
            and (self.fusion_type != 'add')
        if not use_skip:
            conv_modules.append(curr_module)
        else:
            fusion_module = FeatureFusionBlock(
                self.conv_dim, self.output_nc, fusion_type=self.fusion_type,
                norm_fn=self.norm_fn, activation_fn=activation_fn,
                main_module=curr_module)
            conv_modules.append(fusion_module)
        return conv_modules 

    def get_latent_encoder(self):
        feat_res = self.img_size / (2**self.num_layers)
        feat_res /= self.first_conv_stride
        input_dim = (2**3) * self.conv_dim * (feat_res * feat_res)
        layers = [
            Reshape(-1, input_dim),
            nn.Linear(input_dim, self.embed_dim),
            ResLinear(self.embed_dim),
            nn.Linear(self.embed_dim, self.z_dim, bias=True)
        ]
        latent_network = nn.Sequential(*layers)
        return latent_network

    def get_latent_decoder(self):
        input_dim = (2**3) * self.conv_dim
        feat_res = self.img_size / (2**self.num_layers)
        feat_res /= self.first_conv_stride
        layers = [
            nn.Linear(self.z_dim, self.embed_dim),
            ResLinear(self.embed_dim),
            nn.Linear(self.embed_dim, input_dim * feat_res * feat_res),
            Reshape(-1, input_dim, feat_res, feat_res),
        ]
        latent_network = nn.Sequential(*layers)
        return latent_network

    def forward(self, input_var):
        conv_encoder_modules = self.conv_encoder_modules
        conv_decoder_modules = self.conv_decoder_modules
        latent_encoder = self.latent_encoder
        latent_decoder = self.latent_decoder
        ###########################################
        ## Construct the inference (recon) graph ##
        ########################################### 
        enc_features = []
        img_feat = input_var
        for i in xrange(self.num_layers+1):
          img_feat = conv_encoder_modules[i](img_feat)
          enc_features.append(img_feat)
        latent_feat = latent_encoder(img_feat)
        dec_feat = latent_decoder(latent_feat)
        for i in xrange(self.num_layers+1):
          use_skip = (self.skip_start <= i) and (i <= self.skip_end)
          if not use_skip:
              dec_feat = conv_decoder_modules[i](dec_feat)
          elif self.fusion_type == 'add':
              dec_feat = conv_decoder_modules[i](
                  enc_features[self.num_layers - i] + dec_feat)
          else:
              dec_feat = conv_decoder_modules[i](
                  dec_feat, enc_features[self.num_layers - i])
        recon_logit = dec_feat
        recon_prob = self.log_softmax(recon_logit)
        return recon_logit, recon_prob
       
    def get_mode(self):
        return self.eval_mode
 
    def set_mode(self, eval_mode=1):
        if eval_mode:
            for _, v in self.params_dict.iteritems():
                v.eval()
            self.eval_mode = 1
        else:
            for _, v in self.params_dict.iteritems():
                v.train() 
            self.eval_mode = 0
