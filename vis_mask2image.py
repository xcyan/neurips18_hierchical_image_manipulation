### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.mask2image_test_options import MaskToImageTestOptions as TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import sys
from torch.autograd import Variable

opt = TestOptions().parse(save=False, default_args=sys.argv[1:])
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    #generated = model.forward_wrapper(data, True) 
    generated = model.inference(
            label=Variable(data['label']),
            inst=Variable(data['inst']),
            image=Variable(data['image']),
            mask_in=Variable(data['mask_in']),
            mask_out=Variable(data['mask_out'])
            ) 

    visuals = model.get_current_visuals()
    #img_path = data['path']
    print('process image... %s' % ('%05d'% i))
    visualizer.save_images(webpage, visuals, ['%05d' % i])

webpage.save()
