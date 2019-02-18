import torch
from torch.autograd import Variable
from util.util import *
from util.data_util import *
import numpy as np
from PIL import Image
from data.base_dataset import get_transform_params, get_raw_transform_fn, \
                              get_transform_fn, get_soft_bbox, get_masked_image
from options.box2mask_test_options import BoxToMaskTestOptions as MaskGenTestOption
from options.mask2image_test_options import MaskToImageTestOptions as ImgGenTestOption
from models import create_model        
from util.data_util import crop_canvas, paste_canvas

class JointInference():
    def __init__(self, joint_opt):
        ###########################
        # Argument Parsing
        ###########################
        self.opt_maskgen = load_script_to_opt(joint_opt.maskgen_script, MaskGenTestOption)
        self.opt_imggen = load_script_to_opt(joint_opt.imggen_script, ImgGenTestOption)

        # TODO(sh): make this part less hacky
        self.opt_maskgen.gpu_ids = self.opt_imggen.gpu_ids = joint_opt.gpu_ids

        ###########################
        # Model Initialization 
        ###########################
        self.G_box2mask = create_model(self.opt_maskgen)
        self.G_mask2img = create_model(self.opt_imggen)

    def sample_bbox(self, bbox_originals, opt, random=False):
        candidate_list = []
        # sample object based on size
        for bbox in bbox_originals:
            cls = bbox['cls']
            xmin = bbox['bbox'][0]
            ymin = bbox['bbox'][1]
            xmax = bbox['bbox'][2]
            ymax = bbox['bbox'][3]
            box_w, box_h = xmax - xmin, ymax - ymin
            min_axis = min(box_w, box_h)
            max_axis = max(box_w, box_h)
            if max_axis < opt.min_box_size:
                continue
            candidate_list.append(bbox)
        if not random and len(candidate_list) > 0:
            # Sample from bbox within size limit
            return np.random.choice(candidate_list)
        else:
            # Random sample
            return np.random.choice(bbox_originals)

    def sample_window(self, img, label, bbox_sampled):
        pass

    def normalize_input(self, img, label, normalize_image=False):
        tnfm_image_raw = get_raw_transform_fn(normalize=normalize_image)
        tnfm_label_raw = get_raw_transform_fn(normalize=False)
        return tnfm_image_raw(img), tnfm_label_raw(label) * 255.0

    def gen_layout(self, bbox_sampled, label_original, opt):
        # crop canvas
        input_dict = crop_canvas(bbox_sampled, label_original, opt)

        # generate layout
        label_generated = self.G_box2mask.evaluate({
            'label_map': Variable(input_dict['label'], volatile=True),
            'mask_ctx_in': Variable(input_dict['mask_ctx_in'], volatile=True),
            'mask_out': Variable(input_dict['mask_out'], volatile=True),
            'mask_in': Variable(input_dict['mask_in'], volatile=True),
            'cls': Variable(input_dict['cls'], volatile=True),
            'label_map_orig': Variable(input_dict['label_orig'], volatile=True),
            'mask_ctx_in_orig': Variable(input_dict['mask_ctx_in_orig'], volatile=True),
            'mask_out_orig': Variable(input_dict['mask_out_orig'], volatile=True)
        }, target_size=(input_dict['label_orig'].size()[2:4]))

        # paste canvas
        label_canvas = paste_canvas(label_original, label_generated.data, \
            input_dict, resize=False)

        return label_canvas, input_dict, label_generated.data

    def gen_image(self, bbox_sampled, img_original, label_generated, opt):
        if opt.name == 'pix2pixhd_cityscape_512p_noinstance':
            # generate layout
            img_generated = self.G_mask2img.inference(
                Variable(label_generated, volatile=True),
                Variable(torch.zeros_like(label_generated), volatile=True),
                Variable(img_original[0], volatile=True)
            )

            # None are placeholders to match return below
            return img_generated.data, None, None
        else:
            # crop canvas
            input_dict = crop_canvas(bbox_sampled, label_generated, opt, \
                img_original=img_original, transform_img=True)

            # generate layout
            img_generated = self.G_mask2img.inference(
                Variable(input_dict['label'], volatile=True),
                Variable(torch.zeros_like(input_dict['label']), volatile=True),
                Variable(input_dict['image'], volatile=True),
                Variable(input_dict['mask_in'], volatile=True),
                Variable(input_dict['mask_out'], volatile=True)
            )
            # paste canvas
            img_canvas = paste_canvas(img_original, (img_generated.data+1)/2, \
                input_dict, method=Image.BICUBIC, is_img=True)

            return img_canvas, input_dict, img_generated.data
