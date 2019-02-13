"""Factory for box-conditioned mask context auto-encoder."""

import torch
import torch.nn as nn
import torch.nn.parallel
import functools
from torch.autograd import Variable
import numpy as np
from layer_util import *
from MaskContextAE_NET import MaskContextAE_NET


class MaskTwoStreamConvSwitch_NET(MaskContextAE_NET):
    def __init__(self, opt):
        super(MaskTwoStreamConvSwitch_NET, self).__init__(opt)
        # two-stream-specific arguments
        self.which_stream  = opt.which_stream
        assert(('obj' in self.which_stream) or ('context' in self.which_stream))
        
        self.input_nc = opt.label_nc if not (opt.cond_in == 'ctx_obj') \
                else opt.label_nc*2
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.eval_mode = 0
        self.dim_list = [self.conv_dim, 96, 128, 256, 512] # this part is hard-coded
        self.skip_layers = list(range(1,self.num_layers+1))
        self.use_simpleRes = opt.use_simpleRes
        self.n_blocks = opt.n_blocks
        self.add_dilated_layers = opt.add_dilated_layers

    def initialize(self):
        self.conv_encoder_modules = self.get_conv_encoder() # encoder
        self.latent_encoder = self.get_latent_encoder()
        # decoder
        if 'obj' in self.which_stream:
            self.obj_conv_decoder_modules = self.get_conv_decoder(output_nc=1, skip_layers=None)
            self.obj_latent_decoder = self.get_latent_decoder()
        if 'context' in self.which_stream:
            self.ctx_conv_decoder_modules = self.get_conv_decoder(
                    output_nc=self.output_nc, skip_layers=self.skip_layers)
            self.ctx_latent_decoder = self.get_latent_decoder()
        #
        self.params_dict = self.get_params_dict()
        self.trainable_parameters = self.get_trainable_parameters()

    def get_params_dict(self):
        params_dict = dict()
        # shared encoder
        for i, conv_module in enumerate(self.conv_encoder_modules):
            params_dict['conv_encoder_%d' % i] = conv_module
        params_dict['latent_encoder'] = self.latent_encoder
        # obj decoder
        if 'obj' in self.which_stream:
            for i, conv_module in enumerate(self.obj_conv_decoder_modules):
                params_dict['obj_conv_decoder_%d' % i] = conv_module
            params_dict['obj_latent_decoder'] = self.obj_latent_decoder
        if 'context' in self.which_stream:
            for i, conv_module in enumerate(self.ctx_conv_decoder_modules):
                params_dict['ctx_conv_decoder_%d' % i] = conv_module
            params_dict['ctx_latent_decoder'] = self.ctx_latent_decoder
        # context decoder
        return params_dict

    def get_conv_encoder(self):
        activation_fn = nn.ReLU(True)
        output_dim = self.dim_list[0]
        
        layers = []
        layers += [
            nn.Conv2d(self.input_nc, output_dim, stride=2,
            kernel_size=7, padding=3),
            self.norm_fn(output_dim),
            nn.ReLU(),
            ]

        for i in range(self.num_layers):
            input_dim = output_dim
            output_dim = self.dim_list[i+1]
            if self.use_simpleRes:
                layers.append(downResBlock_3x3(
                    input_dim, output_dim,
                    hid_c=input_dim,
                    non_linear=nn.ReLU(True),
                    norm_layer=self.norm_fn))
            else:
                layers.append(ConvResnetBlock(
                        input_dim, output_dim, stride=2,
                        kernel_size=self.conv_size,
                        num_layers=self.num_resnetblocks,
                        norm_fn=self.norm_fn,
                        activation_fn=activation_fn))

        self.latent_dim = output_dim
        return nn.Sequential(*layers)

    def get_latent_encoder(self):
        """ conv encoder """
        activation_fn = nn.ReLU(True)
        #
        layers = []
        if self.add_dilated_layers:
            layers += [DilatedResnetBlock(self.latent_dim, self.latent_dim, dilation=(2,2), activation_fn=self.norm_fn),
                    DilatedResnetBlock(self.latent_dim, self.latent_dim, dilation=(4,4), activation_fn=self.norm_fn)]

        n_blocks = int(np.floor(self.n_blocks/2))
        for i in range(n_blocks): #TODO(sh): change hard-coded numbers
            layers.append(
                ResnetBlock(self.latent_dim,
                    padding_type = 'reflect',
                    norm_layer = self.norm_fn,
                    activation = activation_fn,
                    use_dropout=False))

        return nn.Sequential(*layers)

    def get_latent_decoder(self):
        activation_fn = nn.ReLU(True)
        #
        layers = []
        n_blocks = int(np.ceil(self.n_blocks/2))
        for i in range(n_blocks): #TODO(sh): change hard-coded numbers
            layers.append(
                ResnetBlock(self.latent_dim,
                    padding_type = 'reflect',
                    norm_layer = self.norm_fn,
                    activation = activation_fn,
                    use_dropout=False))

        return nn.Sequential(*layers)

    def get_conv_decoder(self, output_nc, skip_layers=None):
        activation_fn = nn.ReLU(True)
        output_dim = self.latent_dim
        
        layers = []
        for i in range(self.num_layers+1):
            input_dim = output_dim
            if i < self.num_layers:
                output_dim = self.dim_list[self.num_layers-i-1]
            else:
                output_dim = input_dim/2
            if not(skip_layers is None):
                if i in skip_layers: input_dim *= 2
            if self.use_simpleRes:
                layers.append(upResBlock_3x3(
                    input_dim, output_dim,
                    non_linear=nn.ReLU(True),
                    norm_layer=self.norm_fn))
            else:
                layers.append(DeconvResnetBlock(
                        input_dim, output_dim, stride=2,
                        kernel_size=self.conv_size, num_layers=self.num_resnetblocks,
                        norm_fn=self.norm_fn,
                        activation_fn=activation_fn))
        layers.append(
                nn.Conv2d(output_dim, output_nc, kernel_size=3, padding=1, stride=1))

        return nn.Sequential(*layers)

    def forward_decoder(self, decoder, dec_feat, enc_features=None):
        for i, layer in enumerate(decoder):
            if not (enc_features is None) and i in self.skip_layers:
                feat_in = torch.cat((enc_features[-1-self.skip_layers.index(i)],dec_feat),1)
                dec_feat = layer(feat_in)
            else:
                dec_feat = layer(dec_feat)

        return dec_feat

    def forward(self, input_var, cls_onehot, is_bkg=False):
        # cls_onehot: (batch_size, label_nc)
        # input_var: (batch_size, label_nc, img_sz, img_sz)
        shared_conv_encoder_modules = self.conv_encoder_modules
        shared_latent_encoder = self.latent_encoder

        ###########################################
        ## Construct the inference (recon) graph ##
        ###########################################
        # 1. forward through shared conv and latent encoders
        enc_features = []
        enc_feat = input_var
        for i, layer in enumerate(shared_conv_encoder_modules):
            enc_feat = layer(enc_feat)
            if i >= 2 and i < self.num_layers+2: # the first 3 layers are input
                enc_features.append(enc_feat)
        latent_feat = shared_latent_encoder(enc_feat)

        # 2. forward through context conv and latent decoders
        ctx_output_logit, ctx_output_prob = None, None
        if 'context' in self.which_stream:
            ctx_conv_decoder_modules = self.ctx_conv_decoder_modules
            ctx_latent_decoder = self.ctx_latent_decoder
            ctx_dec_feat = ctx_latent_decoder(latent_feat)
            ctx_output_logit = \
                    self.forward_decoder(ctx_conv_decoder_modules, ctx_dec_feat, enc_features)
            ctx_output_prob = self.log_softmax(ctx_output_logit)

        # 3. forward through obj concv and latent decoders
        obj_output_logit, obj_output_prob = None, None
        if 'obj' in self.which_stream:
            obj_conv_decoder_modules = self.obj_conv_decoder_modules
            obj_latent_decoder = self.obj_latent_decoder
            obj_dec_feat = obj_latent_decoder(latent_feat)
            obj_output_logit = \
                    self.forward_decoder(obj_conv_decoder_modules, obj_dec_feat, None)
            obj_output_prob = self.sigmoid(obj_output_logit)

        return ctx_output_logit, ctx_output_prob, obj_output_logit, obj_output_prob

####################################################################
################# UTILIZATION FUNCTION (temporary) #################
####################################################################
class upResBlock_3x3(nn.Module):
    """
        (Residual Blocks + Nearest Upsampling) Block
        We now unify the format of the block thing:
        The 'up' or 'down' block's ratio change is self-defined
        The input has to decide:
        in_c, out_c, hid_c:    defautl setting
                upResBlock: hid_c = in_c
                down~:      hid_c = in_c
                ResBlock:   hid_c = in_c
    """
    def __init__(self, in_c, out_c, hid_c=None, conv2d=None,
            norm_layer=None, non_linear=None):
        super(upResBlock_3x3, self).__init__()

        if hid_c is None: hid_c = in_c;
        if conv2d is None: conv2d = nn.Conv2d;
        if norm_layer is None: norm_layer = Identity;
        if non_linear is None: non_linear = nn.LeakyReLU();

        self.__build_block(in_c, out_c, hid_c, conv2d, norm_layer, non_linear)

    def __build_block(self, in_c, out_c, hid_c, conv2d, norm_layer,
            non_linear):

        self.main_path = nn.Sequential(
                    conv2d(in_c, hid_c, kernel_size=3, padding=1,
                        stride=1),
                    norm_layer(hid_c),
                    non_linear,
                    conv2d(hid_c, out_c, kernel_size=3, padding=1,
                        stride=1),
                )

        self.side_path = conv2d(in_c, out_c, kernel_size=1, padding=0,
                stride=1)

        self.output_layer = nn.Sequential(
                    norm_layer(out_c),
                    non_linear
                )
    def forward(self, input_var):
        input_var = F.upsample(input_var, scale_factor=2,
                mode="bilinear")

        output = self.main_path(input_var)
        res_out = self.side_path(input_var)

        final_output = self.output_layer(res_out + output)

        return final_output

class downResBlock_3x3(nn.Module):
    """
        (Residual Blocks + Conv Downsampling) Block
    """
    def __init__(self, in_c, out_c, hid_c=None, conv2d=None,
            norm_layer=None, non_linear=None):
        super(downResBlock_3x3, self).__init__()

        if hid_c is None: hid_c = in_c;
        if conv2d is None: conv2d = nn.Conv2d;
        if norm_layer is None: norm_layer = Identity;
        if non_linear is None: non_linear = nn.LeakyReLU();

        self.__build_block(in_c, out_c, hid_c, conv2d, norm_layer, non_linear)

    def __build_block(self, in_c, out_c, hid_c, conv2d, norm_layer,
            non_linear):
        self.main_path = nn.Sequential(
                    conv2d(in_c, hid_c, kernel_size=3, padding=1,
                        stride=1),
                    norm_layer(hid_c),
                    non_linear,
                    conv2d(hid_c, out_c, kernel_size=4, padding=1,
                        stride=2),
                )

        self.side_path = conv2d(in_c, out_c, kernel_size=4, padding=1,
                stride=2)

        self.output_layer = nn.Sequential(
                    norm_layer(out_c),
                    non_linear,
                )

    def forward(self, input_var):

        output = self.main_path(input_var)
        res_out = self.side_path(input_var)
        final_output = self.output_layer(res_out + output)

        return final_output

