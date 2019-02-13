"""Training script for learning box-to-mask generation."""

import time
from collections import OrderedDict
from options.box2mask_train_options import BoxToMaskTrainOptions as BaseTrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import sys

opt = BaseTrainOptions().parse(default_args=sys.argv[1:])
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        if (opt.which_epoch is not None) and (opt.which_epoch != 'latest'):
            start_epoch, epoch_iter = int(opt.which_epoch), 0
        else:
            start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
            # TODO(xcyan): fix this hacky implementaton.
            epoch_iter = 0
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
num_batches = int(dataset_size / opt.batchSize)

model = create_model(opt, num_batches)
visualizer = Visualizer(opt)

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % num_batches
    iter_start = epoch_iter
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += 1 #opt.batchSize
        ##################
        ## Forward Pass ##
        ##################
        losses, reconstructed = model.module.forward(
              Variable(data['label']), Variable(data['mask_object_in']),
              Variable(data['mask_context_in']), Variable(data['mask_object_out']),
              Variable(data['mask_out']), 
              Variable(data['mask_object_inst']), 
              Variable(data['cls']), Variable(data['mask_in']), 
              eval_mode=False) 
        comb_recon = reconstructed[0]
        obj_recon = reconstructed[1]

        # TODO(xcyan): double check this.
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        # TODO(sh): add weights for G_recon_comb and G_recon_obj
        if hasattr(model.module, 'kl_decay'):
            kl_weight = opt.kl_weight * (1.0 - (1.0 - opt.kl_start_weight) * model.module.kl_decay)
            loss_G = loss_dict['G_Recon_comb'] + loss_dict['G_Recon_obj'] + \
                    loss_dict['loss_G_GAN'] + \
                    kl_weight * loss_dict['KL_loss']
        else:
            loss_G = loss_dict['G_Recon_comb'] + loss_dict['G_Recon_obj'] + \
                    loss_dict['loss_G_GAN'] 
   
        #####################
        ## Pure generation ##
        #####################
        generated = model.module.generate({
            'label_map': Variable(data['label']),
            'mask_obj_in': Variable(data['mask_object_in']),
            'mask_ctx_in': Variable(data['mask_context_in']),
            'mask_obj_out': Variable(data['mask_object_out']),
            'mask_out': Variable(data['mask_out']),
            'mask_obj_inst': Variable(data['mask_object_inst']),
            'cls': Variable(data['cls']),
            'mask_in': Variable(data['mask_in']) 
            })
        comb_generated = generated['comb_pred_label']
        obj_generated = generated['obj_pred_label']
        #############
        ## Display ##
        #############
        if (total_steps / opt.batchSize) % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if (i == iter_start) or ((total_steps / opt.batchSize) % opt.display_freq == 0):
            visuals = OrderedDict([('gt_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('gt_object_label', util.tensor2im(data['mask_object_inst'][0])),
                                   ('mask_in', util.tensor2im(data['mask_in'][0])),
                                   ('input_context', util.tensor2label(data['mask_context_in'][0], opt.label_nc)),
                                   ('reconstructed_label_comb', util.tensor2label(comb_recon.data[0], opt.label_nc)),
                                   ('reconstructed_label_obj', util.tensor2im(obj_recon.data[0])),
                                   ('comb_generated_label', util.tensor2label(comb_generated.data[0], opt.label_nc)),
                                   ('obj_generated_label', util.tensor2im(obj_generated.data[0]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # Save latest model.
        if (total_steps / opt.batchSize) % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        prev_epoch = epoch - opt.save_epoch_freq * opt.num_checkpoint
        if prev_epoch >= 0:
            model.module.delete_model(prev_epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
  
    ### linearly decay learning rate after certain iterations
    model.module.update_learning_rate(epoch, num_batches)
