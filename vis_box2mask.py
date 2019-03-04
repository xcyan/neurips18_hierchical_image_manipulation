"""Visualization script that performs box-to-mask generation."""

import os
from collections import OrderedDict
from options.box2mask_test_options import BoxToMaskTestOptions as BaseTestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from torch.autograd import Variable
import sys

opt = BaseTestOptions().parse(save=False, default_args=sys.argv[1:])
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.load_image = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
model_dir = os.path.join(opt.checkpoints_dir, opt.name)
assert os.path.isdir(model_dir)
web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'vis_%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    reconstructed = model.reconstruct({
        'label_map': Variable(data['label'], volatile=True),
        'mask_obj_in': Variable(data['mask_object_in'], volatile=True),
        'mask_ctx_in': Variable(data['mask_context_in'], volatile=True),
        'mask_obj_out': Variable(data['mask_object_out'], volatile=True),
        'mask_out': Variable(data['mask_out'], volatile=True),
        'mask_obj_inst': Variable(data['mask_object_inst'], volatile=True),
        'cls': Variable(data['cls'], volatile=True),
        'mask_in': Variable(data['mask_in'], volatile=True)
        }, eval_mode=True)
    reconstructed_comb_label = reconstructed['comb_recon_label']
    reconstructed_obj_label = reconstructed['obj_recon_label']
    #
    generated = model.generate({
        'label_map': Variable(data['label'], volatile=True),
        'mask_obj_in': Variable(data['mask_object_in'], volatile=True),
        'mask_ctx_in': Variable(data['mask_context_in'], volatile=True),
        'mask_obj_out': Variable(data['mask_object_out'], volatile=True),
        'mask_out': Variable(data['mask_out'], volatile=True),
        'mask_obj_inst': Variable(data['mask_object_inst'], volatile=True),
        'cls': Variable(data['cls'], volatile=True),
        'mask_in': Variable(data['mask_in'], volatile=True)
        })
    generated_comb_label = generated['comb_pred_label']
    generated_obj_label = generated['obj_pred_label']

    visuals = OrderedDict([
        ('image', util.tensor2im(data['image'][0])),
        ('gt_label', util.tensor2label(data['label'][0], opt.label_nc)),
        ('input_context', util.tensor2label(data['mask_context_in'][0], opt.label_nc)),
        ('mask_in', util.tensor2im(data['mask_in'][0])),
        ('reconstructed_comb_label', util.tensor2label(reconstructed_comb_label.data[0], opt.label_nc)),
        #('reconstructed_obj_label', util.tensor2im(reconstructed_obj_label.data[0])),
        ('generated_comb_label', util.tensor2label(generated_comb_label.data[0], opt.label_nc)),
        #('generated_obj_label', util.tensor2im(generated_obj_label.data[0]))
    ])
    label_path = data['label_path']
    print('process image... %s' % label_path)
    visualizer.save_images(webpage, visuals, label_path)

webpage.save()
