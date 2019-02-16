"""Testing configurations for mask generation."""

from box2mask_base_options import BoxToMaskOptions

class BoxToMaskTestOptions(BoxToMaskOptions):
    def initialize(self):
        BoxToMaskOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float('inf'))
        self.parser.add_argument('--results_dir', type=str, default='results/')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        self.parser.add_argument('--how_many', type=int, default=50)
        self.parser.add_argument('--num_samples', type=int, default=1)
        self.parser.add_argument('--gendata_dir', type=str, default='gen_ae_512p')
        self.parser.add_argument('--gtdata_dir', type=str, default='gt_512p')
        self.isTrain = False

class JointTestOptions(BoxToMaskOptions):
    def initialize(self):
        BoxToMaskOptions.initialize(self)
        # Add general purpose arguments.
        self.parser.add_argument('--ntest', type=int, default=float('inf'))
        self.parser.add_arugment('--results_dir', type=str, default='results/')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--how_many', type=int, default=50) 
        # Add box2layout arguments.
        self.parser.add_argument('--num_samples', type=int, default=1)
        self.parser.add_argument('--which_epoch', type=str, default='latest')
        # Add layout2img arguments.
        self.parser.add_argument('--pix2pix_name', type=str, default='label2city')
        self.parser.add_argument('--pix2pix_model', type=str, default='CVAE_imggen')
        self.parser.add_argument('--pix2pix_norm', type=str, default='instance')
        self.parser.add_argument('--pix2pix_use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--pix2pix_input_layout', action='store_true', help='input the layout in recognition model')
        
        self.parser.add_argument('--pix2pix_batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--pix2pix_loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--pix2pix_fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--pix2pix_label_nc', type=int, default=35, help='# of input image channels')
        self.parser.add_argument('--pix2pix_output_nc', type=int, default=3, help='# of output image channels')

        self.parser.add_argument('--pix2pix_netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--pix2pix_ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--pix2pix_n_downsample_global', type=int, default=4, help='number of downsampling layers in netG') 
        self.parser.add_argument('--pix2pix_n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--pix2pix_n_blocks_local', type=int, default=3, help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--pix2pix_n_local_enhancers', type=int, default=1, help='number of local enhancers to use')        
        self.parser.add_argument('--pix2pix_niter_fix_global', type=int, default=0, help='number of epochs that we only train the outmost local enhancer')       
        self.parser.add_argument('--pix2pix_instance_feat', action='store_true', help='if specified, add encoded instance features as input')
        self.parser.add_argument('--pix2pix_label_feat', action='store_true', help='if specified, add encoded label features as input')
        self.parser.add_argument('--pix2pix_feat_num', type=int, default=3, help='vector length for encoded features')        
        self.parser.add_argument('--pix2pix_load_features', action='store_true', help='if specified, load precomputed feature maps')
        self.parser.add_argument('--pix2pix_n_downsample_E', type=int, default=3, help='# of downsampling layers in encoder') 
        self.parser.add_argument('--pix2pix_nef', type=int, default=16, help='# of encoder filters in the first conv layer')        
        self.parser.add_argument('--pix2pix_n_clusters', type=int, default=10, help='number of clusters for features')
        self.parser.add_argument('--pix2pix_z_dim', type=int, default=32, help='size of latent vector z')        
        self.parser.add_argument('--pix2pix_z_embed_dim', type=int, default=64, help='size of embedding vector for z')      

        self.isTrain = False
 
