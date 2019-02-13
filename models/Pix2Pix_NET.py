import torch
import torch.nn as nn
from torch.autograd import Variable
from layer_util import *
import numpy as np
import functools

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer='instance', padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.norm_layer = get_norm_layer(norm_layer)

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                self.norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                self.norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=self.norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               self.norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer='instance',
                 padding_type='reflect', use_output_gate=False):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        self.norm_layer = get_norm_layer(norm_layer)
  self.use_output_gate = use_output_gate
        activation = nn.ReLU(True)
  self.input_nc = input_nc
  self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), self.norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      self.norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=self.norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       self.norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, mask=None):
        output = self.model(input)
  if self.use_output_gate and not(mask is None):
    img = input[:,self.input_nc-3:,:,:]
    mask_output = mask.repeat(1, self.output_nc, 1, 1)
    output = (1-mask_output)*img + mask_output*output

  return output

class GlobalTwoStreamGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer='instance',
                 padding_type='reflect', use_skip=False, which_stream='ctx', use_output_gate=False,
                 feat_fusion='early_add', extra_embed=False):
        assert(n_blocks >= 0)
        assert( not (not ('label' in which_stream) and ('late' in feat_fusion)))
        assert( not (not ('ctx' in which_stream) and ('late' in feat_fusion)))
        super(GlobalTwoStreamGenerator, self).__init__()
        self.norm_layer = get_norm_layer(norm_layer)
        self.ngf = ngf
        self.n_downsampling = n_downsampling
        self.padding_type=padding_type
        activation = nn.ReLU(True)
        self.activation = activation
        self.output_nc = output_nc
        self.n_blocks = n_blocks
        self.use_skip = use_skip
        self.which_stream = which_stream
        self.use_output_gate=use_output_gate
        self.feat_fusion = feat_fusion
        feat_dim = self.ngf*2**n_downsampling
        self.extra_embed = extra_embed

        if 'ctx' in which_stream:
            ctx_dim = 3 if not extra_embed else 6
            self.ctx_inputEmbedder = self.get_input(ctx_dim)
            self.ctx_downsampler = self.get_downsampler()
        if 'label' in which_stream:
            self.obj_inputEmbedder = self.get_input(input_nc)
            self.obj_downsampler = self.get_downsampler()
        if which_stream=='ctx_label':
            self.mask_downsampler = nn.MaxPool2d(kernel_size=2**n_downsampling, stride=2**n_downsampling)
            self.feat_fuser = FeatureFusionBlock(feat_dim, feat_dim, feat_fusion.split('_')[1],
                                                self.norm_layer, activation, Identity())
        if 'early' in feat_fusion:
            self.latent_embedder = self.get_embedder(feat_dim, n_blocks)
        elif 'late' in feat_fusion:
            n_blocks_feat_embed = int(np.floor(n_blocks/2.0))
            n_blocks_feat_combine = int(np.ceil(n_blocks/2.0))
            self.obj_latent_embedder = self.get_embedder(feat_dim, n_blocks_feat_embed)
            self.ctx_latent_embedder = self.get_embedder(feat_dim, n_blocks_feat_embed)
            self.latent_embedder = self.get_embedder(feat_dim, n_blocks_feat_combine)
        self.decoder = self.get_upsampler()
        self.outputEmbedder = self.get_output()
        self.feat_dim = feat_dim

    def get_input(self, input_nc):
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, self.ngf, kernel_size=7, padding=0),
                self.norm_layer(self.ngf),
                self.activation]
        return nn.Sequential(*model)

    def get_downsampler(self):
        ### downsample
        model = []
        for i in range(self.n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      self.norm_layer(self.ngf * mult * 2),
                      self.activation]
        return nn.Sequential(*model)

    def get_embedder(self, feat_dim, n_blocks):
        ### resnet blocks
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(feat_dim,
                padding_type=self.padding_type,
                activation=self.activation,
                norm_layer=self.norm_layer)]
        return nn.Sequential(*model)

    def get_upsampler(self):
        ### upsample
        model = []
        for i in range(self.n_downsampling):
            mult = 2**(self.n_downsampling - i)
            dim_in = self.ngf * mult
            dim_out = int(self.ngf * mult / 2)
            if self.use_skip and i > 0:
                dim_in = dim_in*2
            model += [nn.ConvTranspose2d(dim_in, dim_out,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
                       self.norm_layer(int(self.ngf * mult / 2)),
                       self.activation]
        return nn.Sequential(*model)

    def get_output(self):
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0),
                nn.Tanh()]
        return nn.Sequential(*model)

    def forward_encoder(self, inputEmbedder, encoder, input, use_skip):
        enc_feats = []
        enc_feat = inputEmbedder(input)
        for i, layer in enumerate(encoder):
            enc_feat = layer(enc_feat)
            if use_skip and ((i < self.n_downsampling*3-1) and (i % 3 == 2)): # super-duper hard-coded
                enc_feats.append(enc_feat)
        return enc_feat, enc_feats

    def forward_embedder(self, ctx_feat, obj_feat, mask_feat):
        if self.which_stream=='ctx':
            combined_feat = ctx_feat
        elif self.which_stream =='label':
            combined_feat = obj_feat
        elif self.which_stream=='ctx_label':
            if 'late' in self.feat_fusion:
                ctx_feat = self.ctx_latent_embedder(ctx_feat)
                obj_feat = self.obj_latent_embedder(obj_feat)
            ctx_feat = (1-mask_feat)*ctx_feat
            obj_feat = mask_feat*obj_feat
            combined_feat = self.feat_fuser(ctx_feat, obj_feat)
        embed_feat = self.latent_embedder(combined_feat)
        return embed_feat

    def forward_decoder(self, decoder, outputEmbedder, dec_feat, enc_feats):
        for i, layer in enumerate(decoder):
            if (self.use_skip and len(enc_feats) > 0) and ((i > 0) and (i % 3 ==0)): # super-duper hard-coded
                dec_feat = torch.cat((enc_feats[-int((i-3)/3)-1], dec_feat),1)
            dec_feat = layer(dec_feat)
        output = outputEmbedder(dec_feat)
        return output

    def forward(self, img, label, mask):
        ctx_feat = obj_feat = mask_feat = 0
        ctx_feats = []
        if 'ctx' in self.which_stream:
            ctx_feat, ctx_feats = self.forward_encoder(self.ctx_inputEmbedder, self.ctx_downsampler, img, self.use_skip)
        if 'label' in self.which_stream:
            obj_feat, _ = self.forward_encoder(self.obj_inputEmbedder, self.obj_downsampler, label, False)
        if self.which_stream=='ctx_label':
            mask_feat = self.mask_downsampler(mask)
            mask_feat = mask_feat.repeat(1,self.feat_dim,1,1)
        # do embedding
        embed_feat = self.forward_embedder(ctx_feat, obj_feat, mask_feat)
        output = self.forward_decoder(self.decoder, self.outputEmbedder, embed_feat, ctx_feats)
        if self.use_output_gate:
            mask_output = mask.repeat(1, self.output_nc, 1, 1)
            #output = (1-mask_output)*img + mask_output*output
            output = (1-mask_output)*img[:,:3,:,:] + mask_output*output

        return output

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer='instance'):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 self.norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      self.norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       self.norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            indices = (inst == i).nonzero() # n x 4
            for j in range(self.output_nc):
                output_ins = outputs[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]]
                mean_feat = torch.mean(output_ins).expand_as(output_ins)
                outputs_mean[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean
