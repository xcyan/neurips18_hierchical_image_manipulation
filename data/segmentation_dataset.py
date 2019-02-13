### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_transform_params, get_transform_fn, normalize, get_masked_image, get_soft_bbox
from data.base_dataset import get_raw_transform_fn
from data.image_folder import make_dataset
from PIL import Image
import json
import numpy as np
import torch


class SegmentationDataset(BaseDataset):

    def initialize(self, opt): # config=DEFAULT_CONFIG):
        self.opt = opt
        self.root = opt.dataroot
        self.class_of_interest = [] # will define it in child
        self.config = {
            'prob_flip': 0.0 if opt.no_flip else 0.5,
            'prob_bg': opt.prob_bg,
            'fineSize': opt.fineSize,
            'preprocess_option': opt.resize_or_crop,
            'min_box_size': opt.min_box_size,
            'max_box_size': opt.max_box_size,
            'img_to_obj_ratio': opt.contextMargin,
            'patch_to_obj_ratio': 1.2,
            'min_ctx_ratio': 1.2,
            'max_ctx_ratio': 1.5}
        self.check_config(self.config)
        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if (opt.isTrain and (not hasattr(self.opt, 'use_bbox'))) or \
                (hasattr(self.opt, 'load_image') and self.opt.load_image):
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
        self.inst_paths = sorted(make_dataset(self.dir_inst))
        self.dir_bbox = os.path.join(opt.dataroot, opt.phase + '_bbox')
        self.bbox_paths = sorted(make_dataset(self.dir_bbox))

        self.dataset_size = len(self.A_paths)
        self.use_bbox = hasattr(self.opt, 'use_bbox') and (self.opt.use_bbox)
        self.load_image = hasattr(self.opt, 'load_image') and (self.opt.load_image)
        self.load_raw = hasattr(self.opt, 'load_raw') and (self.opt.load_raw)

    def check_config(self, config):
        assert config['preprocess_option'] in ['scale_width', 'none', 'select_region']
        if self.opt.isTrain:
            assert config['img_to_obj_ratio'] < 5.0

    def get_raw_inputs(self, index):
        bbox_path = self.bbox_paths[index]
        with open(bbox_path, 'r') as f:
            inst_info = json.load(f)

        raw_inputs = dict()
        A_path = self.A_paths[index]
        raw_inputs['label'] = Image.open(A_path)
        raw_inputs['label_path'] = A_path

        inst_path = self.inst_paths[index]
        raw_inputs['inst'] = Image.open(inst_path)
        raw_inputs['inst_path'] = inst_path
        if self.load_image:
            B_path = self.B_paths[index]
            raw_inputs['image'] = Image.open(B_path).convert('RGB')
            raw_inputs['image_path'] = B_path
        return raw_inputs, inst_info

    def preprocess_inputs(self, raw_inputs, params):
        outputs = dict()
        # label & inst.
        transform_label = get_transform_fn(self.opt, params, method=Image.NEAREST, normalize=False)
        outputs['label'] = transform_label(raw_inputs['label']) * 255.0
        outputs['inst'] = transform_label(raw_inputs['inst'])
        if self.opt.dataloader == 'sun_rgbd' or self.opt.dataloader == 'ade20k': # NOTE(sh): dirty exception!
            outputs['inst'] *= 255.0
        outputs['label_path'] = raw_inputs['label_path']
        outputs['inst_path'] = raw_inputs['inst_path']
        # image
        if self.load_image:
            transform_image = get_transform_fn(self.opt, params)
            outputs['image'] = transform_image(raw_inputs['image'])
            outputs['image_path'] = raw_inputs['image_path']
        # raw inputs
        if self.load_raw:
            transform_raw = get_raw_transform_fn(normalize=False)
            outputs['label_raw'] = transform_raw(raw_inputs['label']) * 255.0
            outputs['inst_raw'] = transform_raw(raw_inputs['inst'])
            transform_image_raw = get_raw_transform_fn()
            outputs['image_raw'] = transform_image_raw(raw_inputs['image'])
        return outputs

    def preprocess_cropping(self, raw_inputs, outputs, params):
        transform_obj = get_transform_fn(
            self.opt, params, method=Image.NEAREST, normalize=False, is_context=False)
        label_obj = transform_obj(raw_inputs['label']) * 255.0
        input_bbox = np.array(params['bbox_in_context'])
        bbox_cls = params['bbox_cls']
        bbox_cls = bbox_cls if bbox_cls is not None else self.opt.label_nc-1
        mask_object_inst = (outputs['inst']==params['bbox_inst_id']).float() \
                if not (params['bbox_inst_id'] == None) else torch.zeros(outputs['inst'].size())
        ### generate output bbox
        img_size = outputs['label'].size(1) #shape[1]
        context_ratio = np.random.uniform(
          low=self.config['min_ctx_ratio'], high=self.config['max_ctx_ratio'])
        output_bbox = np.array(get_soft_bbox(input_bbox, img_size, img_size, context_ratio))
        mask_in, mask_object_in, mask_context_in = get_masked_image(
            outputs['label'], input_bbox, bbox_cls)
        mask_out, mask_object_out, _ = get_masked_image(
            outputs['label'], output_bbox)
        # Build dictionary
        outputs['input_bbox'] = torch.from_numpy(input_bbox)
        outputs['output_bbox'] = torch.from_numpy(output_bbox)
        outputs['mask_in'] = mask_in # (1x1xHxW)
        outputs['mask_object_in'] = mask_object_in # (1xCxHxW)
        outputs['mask_context_in'] = mask_context_in # (1xCxHxW)
        outputs['mask_out'] = mask_out # (1x1xHxW)
        outputs['mask_object_out'] = mask_object_out # (1xCxHxW)
        outputs['label_obj'] = label_obj
        outputs['mask_object_inst'] = mask_object_inst
        outputs['cls'] = torch.LongTensor([bbox_cls])
        return outputs

    def __getitem__(self, index):
      raw_inputs, inst_info = self.get_raw_inputs(index)
      #
      full_size = raw_inputs['label'].size
      params = get_transform_params(full_size, inst_info,
                                    self.class_of_interest, self.config,
                                    random_crop=self.opt.random_crop)
      outputs = self.preprocess_inputs(raw_inputs, params)
      if self.config['preprocess_option'] == 'select_region':
          outputs = self.preprocess_cropping(raw_inputs, outputs, params)
      return outputs

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SegmentationDataset'



