"""Context AE for mask generation (trainer)."""

import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from base_model import BaseModel
from Discriminator_NET import NLayerDiscriminator, NLayerResDiscriminator, MultiscaleDiscriminator, lr_control 
from mask_losses import MaskReconLoss
from losses import compute_gan_loss, GANLoss

class TwoStreamAE_mask(BaseModel):
    def __init__(self, opt):
        super(TwoStreamAE_mask, self).__init__(opt)
        if opt.resize_or_crop != 'none':
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.which_stream = opt.which_stream
        self.use_gan = opt.use_gan
        self.which_gan = opt.which_gan
        self.gan_weight = opt.gan_weight
        self.rec_weight = opt.rec_weight
        self.cond_in = opt.cond_in
        self.use_output_gate = opt.use_output_gate
        self.opt = opt
        
        if opt.no_comb:
            from MaskTwoStreamConvSwitch_NET import MaskTwoStreamConvSwitch_NET as model_factory
        else:
            from MaskTwoStreamConv_NET import MaskTwoStreamConv_NET as model_factory

        model = self.get_model(model_factory)
        self.netG = model(opt)
        self.netG.initialize()
        # move networsk to gpu
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.netG.cudafy(opt.gpu_ids[0])
 
        print('---------- Networks initialized -------------')
       
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            
            # defaine loss functions
            self.criterionRecon = MaskReconLoss()
            if opt.objReconLoss == 'l1':
                self.criterionObjRecon = nn.L1Loss()
            elif opt.objReconLoss == 'bce':
                self.criterionObjRecon = nn.BCELoss()
            else:
                self.criterionObjRecon = None

            # Names so we can breakout loss
            self.loss_names = ['G_Recon_comb', 'G_Recon_obj', \
                    'KL_loss', 'loss_G_GAN', 'loss_D_GAN', 'loss_G_GAN_Feat']

            params = self.netG.trainable_parameters
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, \
                    betas=(opt.beta1, opt.beta2))
            
            ########## define discriminator
            if self.use_gan:
                label_nc = opt.label_nc if not (opt.cond_in=='ctx_obj') \
                        else opt.label_nc * 2
                if self.which_gan=='patch':
                    use_lsgan=False
                    self.netD = NLayerDiscriminator( \
                            input_nc=1+label_nc, 
                            ndf=opt.ndf,
                            n_layers=opt.num_layers_D,
                            norm_layer=opt.norm_layer,
                            use_sigmoid=not use_lsgan, getIntermFeat=False)
                elif self.which_gan=='patch_res':
                    use_lsgan=False
                    self.netD = NLayerResDiscriminator( \
                            input_nc=1+label_nc, 
                            ndf=opt.ndf,
                            n_layers=opt.num_layers_D,
                            norm_layer=opt.norm_layer,
                            use_sigmoid=not use_lsgan, getIntermFeat=False)
                elif self.which_gan=='patch_multiscale':
                    use_lsgan=True
                    self.netD = MultiscaleDiscriminator(
                            1+label_nc, 
                            opt.ndf, 
                            opt.num_layers_D, 
                            opt.norm_layer, 
                            not use_lsgan, 
                            2, 
                            True)
                self.ganloss = GANLoss(use_lsgan=use_lsgan, 
                        tensor=self.Tensor)
                if opt.use_ganFeat_loss:
                    self.criterionFeat = torch.nn.L1Loss()

                if len(opt.gpu_ids) > 0:
                    self.netD.cuda(opt.gpu_ids[0])
                params_D = [param for param in self.netD.parameters() \
                        if param.requires_grad]
                self.optimizer_D = torch.optim.Adam(
                        params_D, lr=opt.lr, betas=(opt.beta1, 0.999))

            # load networks
            if opt.continue_train or opt.load_pretrain:
                pretrained_path = '' if not self.isTrain else opt.load_pretrain
                self.load_network_dict(
                    self.netG.params_dict, self.optimizer, 'G',
                    opt.which_epoch, opt.load_pretrain)
                if opt.use_gan:
                    # TODO(sh): add loading for discriminator optimizer
                    self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
        else:
            self.load_network_dict(
                self.netG.params_dict, None, 'G', opt.which_epoch, '')

    def name(self):
        return 'TwoStreamAE_mask'

    def get_model(self, model_factory):
        print(self.name())
        return model_factory

    def encode_input(self, label_map, mask_ctx_in, mask_out, mask_in, cls, infer=False):
        if self.opt.label_nc == 3:
            pass
        else:
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            oneHot_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            oneHot_label = oneHot_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            
            oneHot_ctx_in = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            oneHot_ctx_in = oneHot_ctx_in.scatter_(1, mask_ctx_in.data.long().cuda(), 1.0)
            
            oneHot_cls = torch.cuda.FloatTensor(size[0], self.opt.label_nc)
            oneHot_cls = oneHot_cls.scatter_(1, cls.data.long().cuda(), 1.0) 
            
            oneHot_obj_mask_in = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            if not (mask_in is None):
                for b in range(size[0]):
                    oneHot_obj_mask_in[b,cls.data[b,0],:,:] = mask_in.data[b,0]  

        input_label = Variable(oneHot_label)
        input_ctx = Variable(oneHot_ctx_in)
        cls_onehot = Variable(oneHot_cls)
        input_obj_cls_mask = Variable(oneHot_obj_mask_in)
        output_mask = mask_out.cuda()
        return input_label, input_ctx, output_mask, cls_onehot, input_obj_cls_mask

    def discriminate(self, input, cond):
        if 'multiscale' in self.which_gan:
            output = self.netD(torch.cat((input, cond),1))
        else:
            output = self.netD(input, cond)
        return output

    def mask_variable(self, input, mask):
        if not(input.dim()==mask.dim()):
            mask = mask.unsqueeze(1)
        output = input * mask.repeat(1, input.size(1), 1, 1)
        return output

    def forward(self, label_map, mask_obj_in, mask_ctx_in, mask_obj_out, 
            mask_out, mask_obj_inst, cls, mask_in, eval_mode=False):
        output_dict = self.reconstruct(
            {'label_map': label_map,
             'mask_obj_in': mask_obj_in,
             'mask_ctx_in': mask_ctx_in,
             'mask_obj_out': mask_obj_out,
             'mask_out': mask_out,
             'mask_obj_inst': mask_obj_inst, 
             'cls': cls,
             'mask_in': mask_in}, eval_mode=eval_mode)
        comb_recon_label = output_dict['comb_recon_label']
        obj_recon_label = output_dict['obj_recon_label'] 

        # Comput Loss.
        if not eval_mode:
            comb_gt_mask = output_dict['comb_gt_mask']
            obj_gt_mask = output_dict['obj_gt_mask']
            comb_recon_prob = output_dict['comb_recon_prob']
            obj_recon_prob = output_dict['obj_recon_prob']
            label_map = output_dict['label_map']  
            comb_gt_label = label_map.view(-1, label_map.size(2), label_map.size(3)).long()
            loss_recon_comb = loss_recon_obj = 0
            if 'context' in self.which_stream:
                loss_recon_comb = self.criterionRecon(comb_recon_prob, comb_gt_label, comb_gt_mask)
            if 'obj' in self.which_stream and not (self.criterionObjRecon is None):
                if self.use_output_gate:
                    obj_recon_prob = self.mask_variable(obj_recon_prob, comb_gt_mask)
                loss_recon_obj = self.criterionObjRecon(obj_recon_prob, obj_gt_mask)
            # discriminator
            loss_D_GAN = loss_D = loss_G_GAN = 0
            if self.use_gan:
                cond = self.construct_input_cond(output_dict['input_obj_cond'], 
                        output_dict['input_mask'])
                netD_in_real = obj_gt_mask
                netD_in_fake = obj_recon_prob # NOTE(sh): masking twice, but does not matter
                if self.use_output_gate:
                    netD_in_real = self.mask_variable(netD_in_real, comb_gt_mask) 
                    netD_in_fake = self.mask_variable(netD_in_fake, comb_gt_mask) 
                    cond = self.mask_variable(cond, comb_gt_mask) 
                real_digits = self.discriminate(netD_in_real, cond)
                fake_digits = self.discriminate(netD_in_fake.detach(), cond)
                loss_D_real = self.ganloss(real_digits, True)
                loss_D_fake = self.ganloss(fake_digits, False)
                loss_D = 0.5*loss_D_real + 0.5*loss_D_fake
                # GAN feature matching loss
                loss_G_GAN_Feat = Variable(self.Tensor([0]))
                if self.opt.use_ganFeat_loss:
                    feat_weights = 4.0 / (self.opt.num_layers_D + 1)
                    D_weights = 1.0 / 2.0 
                    for i in range(2):
                        for j in range(len(fake_digits[i])-1):
                            loss_G_GAN_Feat += D_weights * feat_weights * \
                                self.criterionFeat(fake_digits[i][j], real_digits[i][j].detach()) * self.opt.lambda_feat
 
                fake_digits = self.discriminate(netD_in_fake, cond)
                loss_G_GAN = self.ganloss(fake_digits, True)

            loss_G = loss_recon_obj + \
                    self.rec_weight * loss_recon_comb + \
                    self.gan_weight * loss_G_GAN 

            g_lr = d_lr = 1.0
            if self.use_gan and self.opt.lr_control:
                g_lr, d_lr = lr_control(loss_G_GAN, loss_D_real, loss_D_fake)

            ##########################
            # Update Generator 
            ##########################
            loss_G = g_lr * loss_G
            self.optimizer.zero_grad()
            loss_G.backward()
            self.optimizer.step()

            ##########################
            # Update Discriminator 
            ##########################
            if self.use_gan:
                loss_D = d_lr * loss_D 
                self.optimizer_D.zero_grad()
                loss_D.backward()
                self.optimizer_D.step()

            loss_kl = 0 # NOTE(sh): I am not using it
            return [loss_recon_comb, loss_recon_obj, loss_kl, loss_G_GAN, loss_D, loss_G_GAN_Feat], \
                    [comb_recon_label, obj_recon_label]
        else:
            return recon_label


    def reconstruct(self, input_dict, eval_mode=False):
        label_map = input_dict['label_map']
        mask_ctx_in = input_dict['mask_ctx_in']
        mask_out = input_dict['mask_out']
        mask_obj_inst = input_dict['mask_obj_inst']
        mask_obj_in = input_dict['mask_obj_in']
        cls = input_dict['cls']
        mask_in = input_dict['mask_in']

        current_mode = self.netG.get_mode()
        if eval_mode != current_mode:
            self.netG.set_mode(eval_mode=eval_mode)
        gt_one_hot, input_ctx, gt_mask, cls_onehot, input_obj_cond = \
              self.encode_input(label_map, mask_ctx_in, mask_out, mask_in, cls)

        cond = self.construct_input_cond(input_obj_cond, input_ctx)
        comb_recon_logit, comb_recon_prob, obj_recon_logit, obj_recon_prob = \
                self.netG.forward(cond, cls_onehot)
        if 'context' in self.which_stream:
            comb_recon_onehot = self.postprocess_output(comb_recon_prob, gt_mask, gt_one_hot)
            _, comb_recon_label = torch.max(comb_recon_onehot, dim=1, keepdim=True) 
        else:
            comb_recon_label = Variable(torch.zeros(gt_mask.size()), volatile=True)
        if not ('obj' in self.which_stream):
            obj_recon_prob = Variable(torch.zeros(mask_obj_inst.size()))
        #
        if eval_mode != current_mode:
            self.netG.set_mode(eval_mode=current_mode)
        output_dict = {'comb_recon_label': comb_recon_label,
                       'obj_recon_label': obj_recon_prob}
        if not eval_mode:
            output_dict.update(
                {'label_map': label_map.cuda(),
                 'input_mask': input_ctx,
                 'comb_gt_mask': gt_mask,
                 'obj_gt_mask': mask_obj_inst.cuda(),
                 'comb_recon_prob': comb_recon_prob,
                 'obj_recon_prob': obj_recon_prob,
                 'input_obj_cond': input_obj_cond})
        return output_dict

    def generate(self, input_dict):
        output_dict = self.reconstruct(input_dict, eval_mode=True)
        return {'comb_pred_label': output_dict['comb_recon_label'],
                'obj_pred_label': output_dict['obj_recon_label']}

    def evaluate(self, input_dict, target_size=None):
        label_map = input_dict['label_map'][0].unsqueeze(0)
        mask_ctx_in = input_dict['mask_ctx_in'][0].unsqueeze(0)
        mask_out = input_dict['mask_out'][0].unsqueeze(0)
        mask_in = input_dict['mask_in'][0].unsqueeze(0)
        cls = input_dict['cls'][0].unsqueeze(0)
        self.netG.set_mode(eval_mode=True)
        
        gt_one_hot, input_ctx, gt_mask, cls_onehot, input_obj_cond = \
              self.encode_input(label_map, mask_ctx_in, mask_out, mask_in, cls)
        cond = self.construct_input_cond(input_obj_cond, input_ctx)
        comb_recon_logit, comb_recon_prob, obj_recon_logit, obj_recon_prob = \
                self.netG.forward(cond, cls_onehot)
        if self.use_output_gate:
            obj_recon_prob  = self.mask_variable(obj_recon_prob, gt_mask)

        if target_size != None:
            label_map = input_dict['label_map_orig'][0].unsqueeze(0).cuda()
            mask_ctx_in = input_dict['mask_ctx_in_orig'][0].unsqueeze(0).cuda()
            mask_out = input_dict['mask_out_orig'][0].unsqueeze(0).cuda()
            gt_one_hot, input_ctx, gt_mask, cls_onehot, input_obj_cond = \
                  self.encode_input(label_map, mask_ctx_in, mask_out, None, cls)
            us = torch.nn.Upsample(target_size, mode='bilinear')
            comb_recon_prob = us(comb_recon_prob)
            obj_recon_prob = us(obj_recon_prob)

        if (cls.data[0,0] == self.opt.label_nc-1):
            comb_recon_onehot = self.postprocess_output(comb_recon_prob, gt_mask, gt_one_hot)
            _, comb_recon_label = torch.max(comb_recon_onehot, dim=1, keepdim=True) 
        else:
            obj_mask = (obj_recon_prob > 0.5).float() # (1, 1, H, W)
            comb_recon_label = (1 - obj_mask) * label_map + obj_mask*cls.data[0,0]
        return comb_recon_label            

    ###############################################################
    ###################### Utility functions ######################
    ###############################################################
    def construct_input_cond(self, obj_cond, ctx_cond):
        if self.cond_in == 'obj':
            cond = obj_cond 
        elif self.cond_in == 'ctx':
            cond = ctx_cond 
        elif self.cond_in == 'ctx_obj':
            cond = torch.cat((obj_cond, ctx_cond),1)
        return cond
    
    # TODO(xcyan): implement blending
    def postprocess_output(self, prob_map, gt_mask, gt_one_hot, use_blending=False):
        try:
            label_map = prob_map * gt_mask + (1 - gt_mask) * gt_one_hot
        except:
            # TODO(xcyan): debug why is gt_mask becomes cpu tensor?
            gt_mask = gt_mask.cuda()
            label_map = prob_map * gt_mask + (1 - gt_mask) * gt_one_hot

        return label_map

    def save(self, which_epoch):
        self.save_network_dict(self.netG.params_dict, 
                self.optimizer, 'G', which_epoch, self.gpu_ids)
        if self.use_gan:
            self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def delete_model(self, which_epoch):
        self.delete_network('G', which_epoch, self.gpu_ids)
        if self.use_gan:
            self.delete_network('D', which_epoch, self.gpu_ids)

    def update_learning_rate(self, epoch=0, data_size=0):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd 
            #for param_group in self.optimizer_D.param_groups:
            #    param_group['lr'] = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            if self.use_gan:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = lr

            print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr

