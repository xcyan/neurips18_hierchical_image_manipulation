import argparse
import os
import sys
import torch
from collections import OrderedDict

from data.segmentation_dataset import SegmentationDataset
from util.visualizer import Visualizer
from util import html
from models.joint_inference_model import JointInference
import util.util as util

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--maskgen_script', type=str,
        default='scripts/vis_box2mask_city.sh',
        help='path to a test script for box2mask generator')
parser.add_argument('--imggen_script', type=str,
        default='scripts/vis_mask2image_city.sh',
        help='path to a test script for mask2img generator')
parser.add_argument('--gpu_ids', type=int,
        default=3,
        help='path to a test script for mask2img generator')
parser.add_argument('--how_many', type=int,
        default=50,
        help='number of examples to visualize')
joint_opt = parser.parse_args()

joint_opt.gpu_ids = [joint_opt.gpu_ids]
joint_inference_model = JointInference(joint_opt)

# Hard-coding some parameters
joint_inference_model.opt_maskgen.load_image = 1
joint_inference_model.opt_maskgen.min_box_size = 128
joint_inference_model.opt_maskgen.max_box_size = -1 # not actually used

opt_maskgen = joint_inference_model.opt_maskgen
opt_pix2pix = joint_inference_model.opt_imggen

# Load data
data_loader = SegmentationDataset()
data_loader.initialize(opt_maskgen)

visualizer = Visualizer(opt_maskgen)
# create website
web_dir = os.path.join('./results', 'test_joint_inference', 'val')
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s' %
                   ('Joint Inference', 'val'))

# Save directory
if not os.path.exists('./results'):
  os.makedirs('./results')
if not os.path.exists('./results/test_joint_inference'):
  os.makedirs('./results/test_joint_inference')
save_dir = './results/test_joint_inference/'

for i in range(data_loader.dataset_size):
  if i >= joint_opt.how_many:
    break

  # Get data
  raw_inputs, inst_info = data_loader.get_raw_inputs(i)
  img_orig, label_orig = joint_inference_model.normalize_input( \
      raw_inputs['image'], raw_inputs['label'], normalize_image=False)
  # Add a dimension
  img_orig = img_orig.unsqueeze(0)
  label_orig = label_orig.unsqueeze(0)
  # List of bboxes
  bboxs = inst_info['objects'].values()

  # Select bbox
  bbox_selected = joint_inference_model.sample_bbox(bboxs, opt_maskgen)
  bbox_selected['cls']=opt_maskgen.label_nc-1
  print(bbox_selected)

  print('generating layout...')
  layout, layout_dict, _ = joint_inference_model.gen_layout(bbox_selected, label_orig, \
      joint_inference_model.G_box2mask, opt_maskgen)

  print('generating image...')
  image, test_dict, img_generated = joint_inference_model.gen_image(bbox_selected, img_orig, \
      layout, joint_inference_model.G_mask2img, opt_pix2pix)

  visuals = OrderedDict([
    ('raw_label', util.tensor2label(label_orig[0], opt_maskgen.label_nc)),
    ('generated_label', util.tensor2label(layout[0], opt_maskgen.label_nc)),
    ('input_label', util.tensor2label(test_dict['label'][0], opt_maskgen.label_nc)),
    ('input_image', util.tensor2im(test_dict['image'][0])),
    ('input_mask', util.tensor2label(test_dict['mask_in'][0], 2)),
    ('label_orig', util.tensor2label(layout_dict['label_orig'][0], opt_maskgen.label_nc)),
    ('mask_ctx_in_orig', util.tensor2label(layout_dict['mask_ctx_in_orig'][0], opt_maskgen.label_nc)),
    ('mask_out_orig', util.tensor2im(layout_dict['mask_out_orig'][0])),
    ('gen_img_patch', util.tensor2im(img_generated[0])),
    ('raw_img', util.tensor2im(img_orig[0], normalize=False)),
    ('generated_img', util.tensor2im(image[0], normalize=False))
  ])
  print('process image... %s' % ('%05d'% i))
  visualizer.save_images(webpage, visuals, ['%05d' % i])

webpage.save()

