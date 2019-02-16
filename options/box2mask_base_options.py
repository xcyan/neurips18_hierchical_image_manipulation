"""Base configuration file for mask generation."""

import argparse
import os
from util import util
import torch

class BoxToMaskOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='box2mask', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--model', type=str, default='AE_maskgen', help='which model to use')
        self.parser.add_argument('--norm_layer', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--use_bbox', type=int, default=1)
        self.parser.add_argument('--load_image', type=int, default=0)

        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=None, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=36, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=36, help='# of output image channels')
        self.parser.add_argument('--contextMargin', type=float, default=2.0, help='')
        self.parser.add_argument('--prob_bg', type=float, default=0.3, help='probablity of sampling random background patches')
        self.parser.add_argument('--min_box_size', type=int, default=32, help='minimum size of the object for sampling')
        self.parser.add_argument('--max_box_size', type=int, default=64, help='maximum size of the sampling box for bkg')
        self.parser.add_argument('--random_crop', type=int, default=1)

        # for setting inputs
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscape/')
        self.parser.add_argument('--dataloader', type=str, default='cityscape')
        self.parser.add_argument('--resize_or_crop', type=str, default='select_region', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # encoder-decoder architecture
        self.parser.add_argument('--conv_dim', type=int, default=64, help='')
        self.parser.add_argument('--conv_size', type=int, default=4, help='')
        self.parser.add_argument('--z_dim', type=int, default=512, help='')
        self.parser.add_argument('--embed_dim', type=int, default=1024, help='')
        self.parser.add_argument('--num_layers', type=int, default=6, help='')
        self.parser.add_argument('--first_conv_stride', type=int, default=1)
        self.parser.add_argument('--first_conv_size', type=int, default=5)
        self.parser.add_argument('--use_resnetblock', type=int, default=1)
        self.parser.add_argument('--num_resnetblocks', type=int, default=1)
        self.parser.add_argument('--fusion_type', type=str, default='add')
        self.parser.add_argument('--skip_start', type=int, default=1)
        self.parser.add_argument('--skip_end', type=int, default=3)

        # for two-stream network
        self.parser.add_argument('--which_stream', type=str, default='obj_context', help='which stream to use on two-stream net (obj|context|obj_context)')
        self.parser.add_argument('--use_gan', action='store_true', help='use gan for two-stream input')
        self.parser.add_argument('--which_gan', type=str, default='patch', help='which gan to use (proj|patch)')
        self.parser.add_argument('--num_layers_D', type=int, default=4, help='number of layers for discirminator')
        self.parser.add_argument('--ndf', type=int, default=64, help='')
        self.parser.add_argument('--objReconLoss', type=str, default='bce', help='loss for reconstruction of object mask (l1,bce,none)')
        self.parser.add_argument('--use_simpleRes', action='store_true', help='loss for reconstruction of object mask (l1,bce,none)')
        self.parser.add_argument('--cond_in', type=str, default='ctx', help='input condition for generator and discriminator (ctx|obj|ctx_obj)')
        self.parser.add_argument('--gan_weight', type=float, default=1.0, help='')
        self.parser.add_argument('--rec_weight', type=float, default=1.0, help='')
        self.parser.add_argument('--lambda_feat', type=float, default=1.0, help='weight for feature matching loss')
        self.parser.add_argument('--n_blocks', type=int, default=4, help='')
        self.parser.add_argument('--no_comb', action='store_true', help='')
        self.parser.add_argument('--add_dilated_layers', action='store_true', help='')

        # for feature-matching network
        self.parser.add_argument('--ngf', type=int, default=64, help='')
        self.parser.add_argument('--n_blocks_gt', type=int, default=4, help='')
        self.parser.add_argument('--n_blocks_masked', type=int, default=4, help='')
        self.parser.add_argument('--n_blocks_decode', type=int, default=4, help='')
        self.parser.add_argument('--use_output_gate', action='store_true', help='if specified, use output gate')

        self.initialized = True

    def parse(self, save=True, default_args=[]):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(default_args)
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save and not self.opt.continue_train:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
