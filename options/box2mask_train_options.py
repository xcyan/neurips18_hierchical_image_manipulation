"""Trainining configurations for mask generation."""

from box2mask_base_options import BoxToMaskOptions

class BoxToMaskTrainOptions(BoxToMaskOptions):
    def initialize(self):
        BoxToMaskOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=40, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=40, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=200, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # checkpointing
        self.parser.add_argument('--num_checkpoint', type=int, default=2)

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='second order momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--enc_lr', type=float, default=1.0)


        # for twostream
        self.parser.add_argument('--lr_control', action='store_true', help='use learning rate control')
        self.parser.add_argument('--mask_gan_input', action='store_true', help='if specified, mask gan input')
        self.parser.add_argument('--use_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')

        self.isTrain = True 
