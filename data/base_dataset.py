### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform_params(full_size, inst_info=None, class_of_interest=None, 
                         config=None, bbox=None, random_crop=True):
    """Prepares the transform parameters (tight object window, soft object window,
        context window, & image window) for cropping.
    
    Args:
        full_size: full image size (tuple of two elements).
        inst_info: instance annotation (dict). 
        class_of_interest: interested class for manipulation (list).
        config: transform configuration (dict).

    Returns (dict):
        crop_pos: image window relative to full image (list).
        crop_pos_object: tight object window to full image (list).
        bbox_in_context: soft object window relative to image window (list).
        bbox_cls: object class.
    """
    flip = random.random() < config['prob_flip']
    orig_w, orig_h = full_size
    target_size = config['fineSize']
    #################
    ## object size ##
    #################
    # if bounding box is not specified, random sample
    if bbox == None:
      ##############
      ## img size ##
      ##############
      min_box_size, max_box_size = config['min_box_size'], config['max_box_size']

      crop_pos, crop_object, bbox_in_context, bbox_cls, bbox_inst_id = \
          crop_single_object(inst_info, class_of_interest, \
          orig_w, orig_h, config['prob_bg'], config['img_to_obj_ratio'],
          config['patch_to_obj_ratio'], min_box_size, max_box_size,
          target_size, flip, random_crop)
    else:
      # use the specified bounding box
      crop_pos, crop_object, bbox_in_context, bbox_cls, bbox_inst_id = \
          crop_single_object_with_bbox(bbox, orig_w, orig_h, \
          config['img_to_obj_ratio'], config['patch_to_obj_ratio'], \
          target_size, random_crop)

    output_dict = {
        'crop_pos': crop_pos,
        'flip': flip,
        'crop_object_pos': crop_object,
        'bbox_in_context': bbox_in_context,
        'bbox_cls': bbox_cls,
        'bbox_inst_id': bbox_inst_id} 
    return output_dict


def crop_single_object_with_bbox(bbox, w, h, img_to_obj_ratio,
        patch_to_obj_ratio, target_size, random_crop=True):
    """ compute cropping region (xmin, ymin, xmax, ymax) for single object """
    # user input bbox
    bbox_cls = bbox['cls']
    bbox_inst_id = 0 # ignore this
    ######################################
    # crop the object and image region
    ######################################
    # set the crop coordinate
    crop_pos = crop_box_with_margin(bbox['bbox'], w, h, img_to_obj_ratio, \
        random_crop)
    crop_object =  crop_box_with_margin(bbox['bbox'], w, h, \
        patch_to_obj_ratio, random_crop)
    if target_size != None:
        bbox_in_context = get_bbox_in_context(bbox, crop_pos, target_size)
    else:
        bbox_in_context = [bbox['bbox'][0]-crop_pos[0], bbox['bbox'][1]-crop_pos[1],
                           bbox['bbox'][0]-crop_pos[0]+bbox['bbox'][2]-bbox['bbox'][0],
                           bbox['bbox'][1]-crop_pos[1]+bbox['bbox'][3]-bbox['bbox'][1]]

    return crop_pos, crop_object, bbox_in_context, bbox_cls, bbox_inst_id

def crop_single_object(inst_info, class_of_interest, 
        w, h, prob_bg, img_to_obj_ratio, patch_to_obj_ratio, min_box_size,
        max_box_size, target_size, flip, random_crop=True):
    """ compute cropping region (xmin, ymin, xmax, ymax) for single object """
    inst_info = inst_info['objects']
    bbox_selected = sample_fg_from_full(inst_info, class_of_interest, min_box_size)
    # sample random box
    sample_bg = random.random() < prob_bg
    if sample_bg or (bbox_selected is None):
        bbox_selected = sample_bg_from_full(min_box_size, max_box_size, w, h) 
    bbox_cls = bbox_selected['cls'] 
    bbox_inst_id = bbox_selected['inst_id']
    ###################################### 
    # crop the object and image region
    ###################################### 
    # set the crop coordinate
    crop_pos = crop_box_with_margin(bbox_selected['bbox'], w, h, \
        img_to_obj_ratio, random_crop)
    crop_object =  crop_box_with_margin(bbox_selected['bbox'], w, h, \
        patch_to_obj_ratio, random_crop)
    bbox_in_context = get_bbox_in_context(bbox_selected, crop_pos, target_size)
    if flip:
        tmp_xmin = bbox_in_context[0]
        tmp_xmax = bbox_in_context[2]
        bbox_in_context[0] = target_size - tmp_xmax
        bbox_in_context[2] = target_size - tmp_xmin

    return crop_pos, crop_object, bbox_in_context, bbox_cls, bbox_inst_id 

def crop_box_with_margin(box, w, h, margin, random_crop=True):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    max_axis = max(xmax-xmin, ymax-ymin)
    ctr = [(xmax + xmin)*0.5, (ymax + ymin)*0.5]
    crop_size = min(max_axis*margin, min(w,h))
    trans_margin =  min(max_axis*(margin-1.0), min(w,h))
    if random_crop:
        crop_xy = [ctr[0] - crop_size*0.5 + (random.random()-0.5)*trans_margin/2.0, \
            ctr[1] - crop_size*0.5 + (random.random()-0.5)*trans_margin/2.0]
    else:
        crop_xy = [ctr[0] - crop_size*0.5, ctr[1] - crop_size*0.5]
    crop_xmin = max(min(max(0, crop_xy[0]), w - crop_size -1),0)
    crop_ymin = max(min(max(0, crop_xy[1]), h - crop_size -1),0)
    crop_xmax = min(crop_xmin+crop_size,w-1)
    crop_ymax = min(crop_ymin+crop_size,h-1)
    crop_coord = [crop_xmin, 
                  crop_ymin, 
                  crop_xmax,
                  crop_ymax]
    return crop_coord


def sample_fg_from_full(inst_info, class_of_interest, min_box_size):
    """Sample one object from the full image.

    Args:
        inst_info: instance annotation (dict).
        class_of_interest: interested class for manipulation (list).
        min_box_size: minimium object size (int).

    Returns:
        bbox_selected: object bounding box with class label (dict),
          containing {'bbox': (xmin, ymin, xmax, ymax), 'cls': cls}.
    """
    candidate_list = []
    # sample object based on size
    for inst_idx in inst_info.keys():
        cls = inst_info[inst_idx]['cls']
        if not (cls in class_of_interest):
            continue
        xmin = inst_info[inst_idx]['bbox'][0]
        ymin = inst_info[inst_idx]['bbox'][1]
        xmax = inst_info[inst_idx]['bbox'][2]
        ymax = inst_info[inst_idx]['bbox'][3]
        box_w, box_h = xmax - xmin, ymax - ymin
        min_axis = min(box_w, box_h) 
        max_axis = max(box_w, box_h)
        #if min_axis < min_box_size:
        if max_axis < min_box_size:
            continue
        # transform box coordinate to center
        candidate_list.append({
            'bbox': [xmin,ymin,xmax,ymax], 'cls': cls, 'inst_id': int(inst_idx)})
    if len(candidate_list) > 0:
        rnd_bbox_idx = np.random.randint(len(candidate_list))
        bbox_selected = candidate_list[rnd_bbox_idx]
        return bbox_selected
    else:
        return None

def sample_bg_from_full(min_box_size, max_box_size, w, h):
  """Sample background object from the full image.
  
  Args:
      min_box_size: minimum size of box (int).
      max_box_size: maximum size of box (int).
      w: width of full image (int).
      h: height of full image (int).

  Returns:
      bbox_selected: object bounding box with class label (dict),
        containing {'bbox': (xmin, ymin, xmax, ymax), 'cls': None}.
  """
  xmin = np.random.randint(0, w-min_box_size-1)
  ymin = np.random.randint(0, h-min_box_size-1)
  xmax = np.random.randint(xmin+min_box_size, min(xmin+max_box_size,w-1))
  ymax = np.random.randint(ymin+min_box_size, min(ymin+max_box_size,h-1))
  bbox_selected = {'bbox': [xmin, ymin, xmax, ymax], 'cls': None, 'inst_id': None}
  return bbox_selected 


def get_bbox_in_context(bbox_selected, crop_pos, target_size):
    """Computes the relative bounding box location in the image (actual image in the training).

    Args:
        bbox_selected: object bounding box with class label (dict),
          containing {'bbox': (xmin, ymin, xmax, ymax), 'cls': cls}.
        crop_pos: adjusted image position (list of four elements).
        target_size: image size (int).

    Returns:
        bbox_in_context: object bounding box location relative to image
          (list of four elements).
    """
    # rescaling.
    xmin = bbox_selected['bbox'][0]
    ymin = bbox_selected['bbox'][1]
    xmax = bbox_selected['bbox'][2]
    ymax = bbox_selected['bbox'][3]
    # transform the object box coordinate according to the cropped window
    x_scale = 1.0 * target_size / (crop_pos[2]-crop_pos[0])
    y_scale = 1.0 * target_size / (crop_pos[3]-crop_pos[1])
    xmin_ = (xmin - crop_pos[0])*x_scale
    ymin_ = (ymin - crop_pos[1])*y_scale
    xmax_ = xmin_ + (xmax-xmin)*x_scale
    ymax_ = ymin_ + (ymax-ymin)*y_scale
    bbox_in_context = [
        max(int(xmin_), 0), 
        max(int(ymin_), 0), 
        min(int(xmax_), target_size), 
        min(int(ymax_), target_size)] 
    return bbox_in_context 

def get_raw_transform_fn(normalize=True):
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_fn(opt, params, method=Image.BICUBIC, normalize=True, is_context=True, resize=True):
    transform_list = []
    # assert opt.resize_or_crop == 'select_region'
    assert opt.resize_or_crop in ['select_region', 'none', 'scale_width']

    if opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
    elif opt.resize_or_crop == 'select_region':
        if is_context:
            crop_pos = params['crop_pos']
        else:
            crop_pos = params['crop_object_pos']
        transform_list.append(transforms.Lambda(lambda img: __select_region(img, crop_pos, opt.fineSize, method, resize)))
    elif opt.resize_or_crop == 'none': # only for testing pix2pixHD
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
            transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def transform_box(opt, params, inst_info):
    def __scale_width_bbox(bbox, ow, oh, target_width):
        resize_factor = 1.0* target_width / ow
        return [b*resize_factor for b in bbox]

    def __scale_minaxis_bbox(bbox, ow, oh, target_axis):
        min_axis = min(ow, oh)
        resize_factor = 1.0*target_axis / min_axis
        return [b*resize_factor for b in bbox]

    def __crop_bbox(bbox, crop_pos, target_size):
        cx,cy = crop_pos
        if bbox[2] <= cx or bbox[3] <= cy or \
                bbox[0] >= cx + target_size or bbox[1] >= cy + target_size:
            bbox=None
        else:
            bbox[0] = max(bbox[0]-cx,0)
            bbox[1] = max(bbox[1]-cy,0)
            bbox[2] = min(bbox[2]-cx,target_size-1)
            bbox[3] = min(bbox[3]-cy,target_size-1)
        return bbox

    def __flip_bbox(bbox, target_width):
        return [target_width-bbox[2],bbox[1],target_width-bbox[0],bbox[3]]

    # extract bbox
    oh,ow = inst_info['imgHeight'], inst_info['imgWidth']
    transformed_objs = {}
    iids = inst_info['objects'].keys()
    for iid in iids:
        bbox = inst_info['objects'][iid]['bbox'] # [xmin, ymin, xmax, ymax]
        if 'scale_width' in opt.resize_or_crop:
            bbox = __scale_width_bbox(bbox, ow, oh, opt.loadSize)
            target_size = opt.loadSize
        elif 'scale_minaxis' in opt.resize_or_crop:
            bbox = __scale_minaxis_bbox(bbox, ow, oh, opt.loadSize)
            target_size = opt.loadSize
        if 'crop' in opt.resize_or_crop:
            bbox = __crop_bbox(bbox, params['crop_pos'], opt.fineSize)
            target_size = opt.fineSize
            if bbox==None: 
                continue
            if bbox[2]-bbox[0] < 1 or bbox[3]-bbox[1] < 1:
                continue
        if params['flip']:
            bbox = __flip_bbox(bbox, target_size)
        transformed_objs[iid]={'bbox':bbox, 'cls':inst_info['objects'][iid]['cls']}
    return transformed_objs

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# input_tuple = [w_lo, h_lo, w_hi, h_hi].
def get_soft_bbox(input_tuple, ow, oh, ratio=1.5):
    w_len = input_tuple[2] - input_tuple[0]
    h_len = input_tuple[3] - input_tuple[1]
    w_center = (input_tuple[0] + input_tuple[2]) / 2
    h_center = (input_tuple[1] + input_tuple[3]) / 2
    w_len *= ratio
    h_len *= ratio
    output_tuple = [
        max(int(w_center - w_len / 2), 0),
        max(int(h_center - h_len / 2), 0),
        min(int(w_center + w_len / 2), ow),
        min(int(h_center + h_len / 2), oh)]
    return output_tuple


def get_masked_image(image_tensor, bbox_tensor, cls2fill=0):
    masked_tensor = torch.zeros(1, image_tensor.size(1), image_tensor.size(2))
    wmin, hmin, wmax, hmax = int(bbox_tensor[0]), int(bbox_tensor[1]), \
        int(bbox_tensor[2]), int(bbox_tensor[3])
    if hmax > hmin and wmax > wmin: 
        try:
            masked_tensor[0, hmin:hmax, wmin:wmax] = 1
        except:
            print('Exception in get_masked_image')
            print(masked_tensor)
            print('%d %d %d %d' % (hmin, hmax, wmin, wmax))
            print(masked_tensor.size())
    masked_object = masked_tensor * image_tensor
    masked_context = (1 - masked_tensor) * image_tensor + \
            masked_tensor * cls2fill
    return masked_tensor, masked_object, masked_context

#
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __scale_minaxis(img, target_axis, method=Image.BICUBIC):
    ow, oh = img.size
    min_axis = min(ow, oh)
    w = int(target_axis * ow / min_axis)
    h = int(target_axis * oh / min_axis)    
    return img.resize((w, h), method)

def __select_region(img, crop_pos, target_size, method=Image.BICUBIC, resize=True): # is_context=True):
    img = img.crop((crop_pos[0], crop_pos[1], crop_pos[2], crop_pos[3])) #crop
    if resize:
        img = img.resize((target_size, target_size), method) # resize
    return img

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

