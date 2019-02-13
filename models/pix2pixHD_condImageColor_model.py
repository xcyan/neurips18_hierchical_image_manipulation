"""
Pix2pixHD model with additional image data as input
It basically takes additional inputs of cropped image, which is then used as
additional 3-channel input.
The additional image input has "hole" in the regions inside object bounding box,
which is subsequently filled by the generator network.
"""
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from layer_util import *
import util.util as util
from collections import OrderedDict

NULLVAL = 0.0

# TODO(sh): change the forward_wrapper things to enable multi-gpu training
# TODO(sh): add additional input of "mask_in" for the binary mask of object region
# TODO(sh): enlarge the context marign
class Pix2PixHDModel_condImgColor(BaseModel):
    def __init__(self, opt):
        super(Pix2PixHDModel_condImgColor, self).__init__(opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.netG_type = opt.netG
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        # NOTE(sh): 3-channels for adddional rgb-image
        input_nc = opt.label_nc if opt.label_nc != 0 else 3

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num
        from .Pix2Pix_NET import GlobalGenerator, GlobalTwoStreamGenerator
        if opt.netG=='global':
            netG_input_nc += 3
            self.netG = GlobalGenerator(netG_input_nc, opt.output_nc, opt.ngf,
                    opt.n_downsample_global, opt.n_blocks_global, opt.norm, 'reflect', opt.use_output_gate)
        elif opt.netG=='global_twostream':
            self.netG = GlobalTwoStreamGenerator(netG_input_nc, opt.output_nc, opt.ngf,
                    opt.n_downsample_global, opt.n_blocks_global, opt.norm, 'reflect', opt.use_skip,
                    opt.which_encoder, opt.use_output_gate, opt.feat_fusion, True)
        else:
            raise NameError('global generator name is not defined properly: %s' % opt.netG)
        print(self.netG)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.cuda(opt.gpu_ids[0])
        self.netG.apply(weights_init)

        # Discriminator network
        if self.isTrain:
            self.no_imgCond = opt.no_imgCond
            self.mask_gan_input = opt.mask_gan_input
            self.use_soft_mask = opt.use_soft_mask
            use_sigmoid = opt.no_lsgan
            if self.no_imgCond:
                netD_input_nc = input_nc + opt.output_nc
            else:
                netD_input_nc = input_nc + 3 + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            if opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
                netD_input_nc = 3
            from .Discriminator_NET import MultiscaleDiscriminator
            self.netD = MultiscaleDiscriminator(netD_input_nc, opt.ndf, opt.n_layers_D,
                    opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss)
            print(self.netD)
            if len(opt.gpu_ids) > 0:
                assert(torch.cuda.is_available())
                self.netD.cuda(opt.gpu_ids[0])
            self.netD.apply(weights_init)

        ### Encoder network
        if self.gen_features:
            from .Pix2Pix_NET import Encoder
            self.netE = Encoder(opt.output_nc, opt.feat_num, opt.nef, opt.n_downsample_E, opt.norm)
            print(self.netE)
            if len(opt.gpu_ids) > 0:
                assert(torch.cuda.is_available())
                self.netE.cuda(opt.gpu_ids[0])
            self.netE.apply(weights_init)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            from .losses import GANLoss, VGGLoss
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG.parameters())
            if self.gen_features:
                params += list(self.netE.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def name(self):
        return 'Pix2PixHDModel_condImg'

    def encode_instwise_embedding(self, inst_map, embedding):
        pass

    def encode_global_embedding(self, mask_in, embedding):
        """
        Encode the embedding globally to the mask region
        Input:
            embedding: (B, K)
            mask_in: (B, 1, H, W)
        Output:
            embeded_tensor: (B, K, H, W)
        """
        B, K = embedding.size(0), embedding.size(1)
        H, W = mask_in.size(2), mask_in.size(3)
        embeded_tensor = embedding.view(B,K,1,1).repeat(1,1,H,W)
        embeded_tensor = embeded_tensor * mask_in.expand_as(embeded_tensor)
        return embeded_tensor

    def get_color_embedding(self, inst_map, image):
        """
        get the color embedding from the input object instance
        Input:
            inst_map: (B, 1, H, W)
            image: (B, 3, H, W)
        Output:
            embedding: (B,3)
        """
        size = image.size()
        masked_image = image * inst_map.expand_as(image)
        #embedding = masked_image.view(size[0], size[1], -1).mean(dim=2)
        embedding = masked_image.view(size[0], size[1], -1).sum(dim=2)
        noise = torch.cuda.FloatTensor(size[0],size[1])
        noise = noise.uniform_(0,1)*0.06 + 0.97
        for b in range(size[0]):
            if inst_map[b].data.sum() > 0:
                embedding[b].data /= inst_map[b].data.sum()
            else:
                embedding[b].data.fill_(0.0)
        # add noise
        embedding *= Variable(noise)
        embedding[embedding<-1]=-1
        embedding[embedding>1]=1
        return embedding

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, mask_in=None, obj_mask=None, color_embed=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        assert(real_image is not None)
        assert(mask_in is not None)
        real_image = Variable(real_image.data.cuda())
        mask_object_box = mask_in.repeat(1,3,1,1).cuda()
        cond_image = (1 - mask_object_box) * real_image + mask_object_box * NULLVAL # TODO(sh): define null_img

        # get color embedding
        if not infer or color_embed is None:
            color_embed = self.get_color_embedding(obj_mask.cuda(), real_image)
        else:
            #color_embed = color_embed * obj_mask.data.mean() # this is the temperally patch
            color_embed = Variable(color_embed.cuda())
        color_embed_cond = self.encode_global_embedding(mask_in.cuda(), color_embed)

        cond_image = torch.cat((cond_image, color_embed_cond), 1)

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map, cond_image

    def discriminate(self, input_label, test_image, mask, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if self.opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
            input_concat = test_image.detach()
        if self.mask_gan_input:
            input_concat = input_concat * mask.repeat(1, input_concat.size(1), 1, 1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward_wrapper(self, data, infer=False):
        label = Variable(data['label'])
        inst = Variable(data['inst'])
        image = Variable(data['image'])
        mask_in = Variable(data['mask_in'])
        mask_out = Variable(data['mask_out'])
        feat = None
        losses, generated = self.forward(label, inst, image, feat, mask_in, mask_out, infer)
        return losses, generated

    def forward(self, label, inst, image, feat, mask_in, mask_out, obj_mask, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map, cond_image = self.encode_input(label, inst, image, feat, mask_in=mask_in, obj_mask=obj_mask)

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_label = torch.cat((input_label, cond_image), 1)
        # Fake Generation
        input_concat = input_label
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
        elif self.netG_type == 'global_twostream':
            fake_image = self.netG.forward(cond_image, input_mask, mask_in)

        # Fake Detection and Loss
        if self.no_imgCond:
            netD_cond = input_mask
        else:
            netD_cond = input_label
        mask_cond = mask_in if not self.use_soft_mask else mask_out
        pred_fake_pool = self.discriminate(netD_cond, fake_image, mask_cond, True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(netD_cond, real_image, mask_cond, False)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        netD_in = torch.cat((netD_cond, fake_image), dim=1)
        if self.opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
            netD_in = fake_image
        if self.mask_gan_input:
            netD_in = netD_in * mask_cond.repeat(1, netD_in.size(1), 1, 1)
        pred_fake = self.netD.forward(netD_in)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = Variable(self.Tensor([0]))
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = Variable(self.Tensor([0]))
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        # color matching loss
        if self.opt.lambda_rec > 0:
            # TOOD(sh): this part is bit hacky but let's leave it for now
            loss_G_GAN_Feat += self.criterionFeat(fake_image, real_image.detach()) * self.opt.lambda_rec

        self.fake_image = fake_image.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_image = cond_image.cpu().data[0][:3,:,:]

        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ], None if not infer else fake_image ]

    def inference(self, label, inst, image, mask_in, color_embed, obj_mask):
        # Encode Inputs
        input_label, inst_map, real_image, _, cond_image = self.encode_input(label, inst, image, mask_in=mask_in, obj_mask=obj_mask, color_embed=color_embed, infer=True)
        mask_in = mask_in.cuda()

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_label = torch.cat((input_label, cond_image), 1)

        # Fake Generation
        input_concat = input_label
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
        elif self.netG_type == 'global_twostream':
            mask_in = mask_in.cuda()
            fake_image = self.netG.forward(cond_image, input_mask, mask_in)

        self.fake_image = fake_image.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_image = cond_image.cpu().data[0][:3,:,:]
        self.input_color = cond_image.cpu().data[0][3:,:,:]

        return fake_image

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def get_current_visuals(self):
        return OrderedDict([
            ('input_label', util.tensor2label(self.input_label, self.opt.label_nc)),
            ('input_image', util.tensor2im(self.input_image)),
            ('real_image', util.tensor2im(self.real_image)),
            ('synthesized_image', util.tensor2im(self.fake_image))
            ])

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def delete_model(self, which_epoch):
        self.delete_network('G', which_epoch, self.gpu_ids)
        self.delete_network('D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
