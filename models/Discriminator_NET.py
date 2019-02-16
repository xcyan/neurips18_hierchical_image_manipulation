import torch
import torch.nn as nn
from torch.autograd import Variable
from layer_util import *
import numpy as np
import functools

##############################################################################
# Multi-Scale Discriminator 
##############################################################################
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer='instance', 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.norm_layer = get_norm_layer(norm_layer)
     
        for i in range(num_D):
            netD = NLayerDiscriminator(
                    input_nc, ndf, n_layers, 
                    norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), 
                            getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(
                3, stride=2, padding=[1, 1], count_include_pad=False)

        self.apply(weights_init)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) 
                        for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
            norm_layer='instance', use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.norm_layer = get_norm_layer(norm_layer)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), 
            nn.LeakyReLU(0.2, True)
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                self.norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            self.norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        self.apply(weights_init)

    def forward(self, input, cond):
        if not cond is None:
            input = torch.cat((input,cond),1)
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerResDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
            norm_layer='instance', use_sigmoid=False, getIntermFeat=False,
            num_resnetblocks=1):
        super(NLayerResDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.norm_layer = get_norm_layer(norm_layer)
        self.num_resnetblocks = num_resnetblocks

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[
            ConvResnetBlock(
                    input_nc, ndf, stride=2,
                    kernel_size=kw, 
                    num_layers=self.num_resnetblocks,
                    norm_fn=self.norm_layer,
                    activation_fn=nn.LeakyReLU(0.2,True))
        ]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                ConvResnetBlock(
                        nf_prev, nf, stride=2,
                        kernel_size=kw, 
                        num_layers=self.num_resnetblocks,
                        norm_fn=self.norm_layer,
                        activation_fn=nn.LeakyReLU(0.2,True))
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            self.norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        self.apply(weights_init)

    def forward(self, input, cond):
        input = torch.cat((input,cond),1)
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        


def lr_control(loss_G, loss_D_real, loss_D_fake, gan_margin=0.3):
    update_g = True
    update_d = True
    if loss_D_real.data[0] < gan_margin or loss_D_fake.data[0] < gan_margin:
        update_d = False
    if loss_D_real.data[0] > (1 - gan_margin) or loss_D_fake.data[0] > (1 - gan_margin):
        update_g = False
    if not (update_d or update_g):
        update_d = True
        update_g = True
    g_lr = float(update_g)
    d_lr = float(update_d)
    if not update_g:
        print('Froze Generator\t[G=%.3f],[DR=%.3f],[DF=%.3f]' % \
                (loss_G.data[0], loss_D_real.data[0], loss_D_fake.data[0]))
    elif not update_d:
        print('Froze Discriminator\t[G=%.3f],[DR=%.3f],[DF=%.3f]' % \
                (loss_G.data[0], loss_D_real.data[0], loss_D_fake.data[0]))
    else:
        print('Update Both\t[G=%.3f],[DR=%.3f],[DF=%.3f]' % \
                (loss_G.data[0], loss_D_real.data[0], loss_D_fake.data[0]))
    return g_lr, d_lr
##############################################################################
# Utils for Discriminator with Projection Layer + Spectral Normalization
##############################################################################
# TODO(sh): I temperolly added util blocks to here, but will move it to 
# layer_utils.py later if this model works
def conv3x3(in_c, out_c, kernel_size=3, stride=1, padding=1, conv2d=None):
    """
        only a single layer, no non-linear and no normalization layer
    """
    if conv2d is None:
        conv2d = nn.Conv2d

    conv_layer = conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, 
            padding=padding, bias=False)
    return conv_layer

    
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
        if non_linear is None: non_linear = nn.LeakyReLU;

        self.__build_block(in_c, out_c, hid_c, conv2d, norm_layer, non_linear)

    def __build_block(self, in_c, out_c, hid_c, conv2d, norm_layer, 
            non_linear):
        self.main_path = nn.Sequential(
                    conv2d(in_c, hid_c, kernel_size=3, padding=1, 
                        stride=1),
                    norm_layer(hid_c),
                    non_linear(),
                    conv2d(hid_c, out_c, kernel_size=4, padding=1, 
                        stride=2),
                )

        self.side_path = conv2d(in_c, out_c, kernel_size=4, padding=1,
                stride=2)

        self.output_layer = nn.Sequential(
                    norm_layer(out_c),
                    non_linear(),
                )

    def forward(self, input_var):

        output = self.main_path(input_var)
        res_out = self.side_path(input_var)
        final_output = self.output_layer(res_out + output)

        return final_output

class ResLinear(nn.Module):
    """Residual block with fully-connected connections"""
    def __init__(self, in_c, out_c, hid_c=None, linear_layer=None, 
            norm_layer=None, non_linear=None):
        super(ResLinear, self).__init__()
        if hid_c is None: hid_c = in_c;
        if norm_layer is None: norm_layer = Identity;
        if non_linear is None: non_linear = nn.LeakyReLU;
        if linear_layer is None: linear_layer = nn.Linear

        self.__build_block(in_c, out_c, hid_c, linear_layer, norm_layer, 
                non_linear)

    def __build_block(self, in_c, out_c, hid_c, linear_layer, norm_layer, 
            non_linear):

        self.deep_connect = nn.Sequential(
                    linear_layer(in_c, hid_c),
                    norm_layer(hid_c),
                    non_linear(),
                    linear_layer(hid_c, out_c),
                )

        self.shortcut_connect = linear_layer(in_c, out_c)

        self.output_layer = nn.Sequential(
                norm_layer(out_c),
                non_linear()
                )
            

    def forward(self, input_var):
        output = self.shortcut_connect(input_var) + self.deep_connect(input_var)
        output = self.output_layer(output)
        
        return output

class ResBlock(nn.Module):
    """
    Residual block with convolution filters
    """

    def __init__(self, in_c, out_c=None, hid_c=None, conv2d=None,
            final_activ=None, norm_layer=None, non_linear=None, 
            kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        if out_c is None: out_c = in_c;
        if hid_c is None: hid_c = out_c;
        if norm_layer is None: norm_layer = Identity;
        if non_linear is None: non_linear = nn.LeakyReLU
        if final_activ is None: final_activ = nn.LeakyReLU

        self.block = nn.Sequential(
            conv3x3(in_c, hid_c, kernel_size=kernel_size, 
                stride=stride, padding=padding, conv2d=conv2d),
            norm_layer(hid_c),
            non_linear(),
            conv3x3(hid_c, out_c, kernel_size=kernel_size, 
                stride=stride, padding=padding, conv2d=conv2d),
            )
        if out_c != in_c:
            self.side_block = conv3x3(in_c, out_c, kernel_size=kernel_size,
                    stride=stride, padding=padding, conv2d=conv2d)
        else:
            self.side_block = None

        self.final_activ = final_activ()
        self.output_norm = norm_layer(out_c)


    def forward(self, x):
        residual = x
        if self.side_block is not None:
            residual = self.side_block(residual)

        out = self.block(x)
        out += residual
        out = self.output_norm(out)
        out = self.final_activ(out)
        return out

class SideBlock(nn.Module):
    def __init__(self, in_c, out_c, conv2d=None, norm_layer=None, 
            kernel_size=3, padding=1, stride=1, non_linear=nn.ReLU):
        super(SideBlock, self).__init__()
        if conv2d is None: conv2d = nn.Conv2d;
        if norm_layer is None: norm_layer = Identity;
        if non_linear is None: non_linear = Identity;

        self.main_path = nn.Sequential(
                   conv2d(in_c, out_c, kernel_size=kernel_size,
                       padding=padding, stride=stride),
                   norm_layer(out_c),
                )

        self.non_linear = non_linear()

    def forward(self, input_var):
        output_var = self.main_path(input_var)
        return self.non_linear(output_var), output_var

def switch_grad(net, switch_flag):
    """
        for '.u' we always want the requires_grad to be False, because 
        the update is not got by gradients descent.
        Instead it is got by power iteration.
    """
    for key, var in net.state_dict(keep_vars=True).items():
        if not key.endswith(".u"):
            var.requires_grad = switch_flag

##############################################################################
# Discriminator with Projection Layer + Spectral Normalization
##############################################################################
# TODO(sh): update hard-coded dims
from sn_utils import *
class Res_Discriminator(nn.Module):
    """
        ResNet Like Discriminator with condition
        Architecture is just like Paper: "cGANs WITH PROJECTION DISCRIMINATOR" 
        Fig 14(a). But No Global Sum Pooling because I am worried about the
        the model's capacity to dig out the distribution in text description.
        (If having memory issue, we should revise this maybe)
    """
    def __init__(self, ndf, n_layers, input_nc, label_nc):
        super(Res_Discriminator, self).__init__()
        self.ndf = ndf # 64
        self.downsample_num = n_layers # 4
        self.input_nc = input_nc
        self.label_nc = label_nc

        self.downResBlock_3x3 = lambda x: downResBlock_3x3(*x, conv2d=SNConv2d)
        self.ResLinear = lambda x, y: ResLinear(*x, non_linear=nn.LeakyReLU, 
                norm_layer=None, linear_layer=SNLinear, **y)
        self.ResBlock = lambda x: ResBlock(*x, conv2d=SNConv2d, 
                final_activ=nn.ReLU, non_linear=nn.ReLU)
        #self.SideBlock = lambda x, y: SideBlock(*x, conv2d=SNConv2d, **y)

        self.__build_model()

    def __build_model(self):
        main_head_list = []
        side_path_list = []
        mask_embe_list = []
        main_head_list.append(self.downResBlock_3x3((self.input_nc, self.ndf)))

        input_c = self.ndf # 64
        dim_list = [96, 128, 256, 512] # TODO(sh): modfiy the hand-coded part 
        for i in range(self.downsample_num):
            main_head_list.append(self.downResBlock_3x3((input_c , 
                dim_list[i])))

            input_c = dim_list[i] # final 256, 16 x 32 (DOWN_NUM_D + 1) 
                        # if the image size is 256 x 512

        mask_embedding_layer = []
        self.mask_embedding_layer = nn.Sequential(
            conv3x3(self.label_nc, 128, 
                kernel_size=7, padding=3, stride=2, conv2d=SNConv2d), 
            nn.LeakyReLU(),
            conv3x3(128, 256, 
                kernel_size=7, padding=3, stride=2, conv2d=SNConv2d), 
            nn.LeakyReLU(),
            conv3x3(256, 512, 
                kernel_size=7, padding=3, stride=2, conv2d=SNConv2d), 
            nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        )

        #side_path_list.append(self.SideBlock((input_c, 512), {})) 

        self.main_head = nn.Sequential(*main_head_list)
        self.side_path = nn.Sequential(*side_path_list)

        self.forward_to_scalar = nn.Sequential(
                   self.ResBlock((input_c, )),
                   SumPooling(),
                   SNLinear(input_c, 1)
                )
        """
        self.forward_to_scalar = nn.Sequential(
                   self.ResBlock((input_c, )),
                   self.downResBlock_3x3((input_c, input_c * 2)),
                   self.downResBlock_3x3((input_c * 2, input_c * 4)),
                   SumPooling(),
                   SNLinear(input_c * 4, 1)
                )
        """

    def forward(self, input_var, mask_var):
        """
            The notation here is kind of messy:
            Basically, it means:
            1. mask_var: the very raw binary coded dense label information
            2. mask_var_next: the flows that continues to donwside the projected
                conditioned label information by embedding
            3. mask_embedded: difference between this and mask_var_next is that
                it does not take the final nonlinear activation

            input_var ----> input_var ----> input_var ---> ... ---> x_scalar
                                |               |
                                |               |
                            input_side      input_side  

                            (inner product) ---> a scalar of a certain
                               |                 scale
                               |
            mask_var ---> mask_embedd/mask_var_next ---> mask_embedd/mask_var_next
        """
        batch_size = input_var.size(0)
        input_inter = self.main_head(input_var)

        mask_embedded = self.mask_embedding_layer(mask_var)

        #input_side, _ = self.side_path(input_inter)
        #project_scalar = torch.bmm(input_side.view(batch_size, 1, -1), 
        #        mask_embedded.view(batch_size, -1, 1))
        project_scalar = torch.bmm(input_inter.view(batch_size, 1, -1), 
                mask_embedded.view(batch_size, -1, 1))

        x_scalar = self.forward_to_scalar(input_inter)

        output_scalar = project_scalar.view(-1, 1) + x_scalar

        return output_scalar
