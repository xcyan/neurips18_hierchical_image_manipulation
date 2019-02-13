from PIL import Image 
import numpy as np
import torch
from data.base_dataset import get_transform_params, get_raw_transform_fn, \
                              get_transform_fn, get_soft_bbox, get_masked_image
from .util import *

def crop_canvas(bbox_sampled, label_original, opt, img_original=None, \
    patch_to_obj_ratio=1.2, min_ctx_ratio=1.2, max_ctx_ratio=1.5, resize=True, \
    transform_img=False):
    h, w = label_original.size()[2:4]

    config = {}
    config['prob_flip'] = 0.0
    config['fineSize'] = opt.fineSize if resize else None
    config['img_to_obj_ratio'] = opt.contextMargin
    config['patch_to_obj_ratio'] = patch_to_obj_ratio
    config['min_ctx_ratio'] = min_ctx_ratio
    config['max_ctx_ratio'] = max_ctx_ratio

    params = get_transform_params((w,h), config=config, bbox=bbox_sampled, \
        random_crop=False)
    transform_label = get_transform_fn(opt, params, method=Image.NEAREST, \
        normalize=False, resize=resize)
    transform_image = get_transform_fn(opt, params, resize=resize)

    output_dict = {}
    output_dict['label'] = transform_label(tensor2pil(label_original[0])) * 255.0
    if transform_img:
        output_dict['image'] = transform_image(tensor2pil(img_original[0], \
            is_img=True))

    input_bbox = np.array(params['bbox_in_context'])
    crop_pos = np.array(params['crop_pos']).astype(int)
    bbox_cls = params['bbox_cls']
    ### generate output bbox
    img_size = output_dict['label'].size(1) #shape[1]
    context_ratio = np.random.uniform(low=config['min_ctx_ratio'],\
         high=config['max_ctx_ratio'])
    output_bbox = np.array(get_soft_bbox(input_bbox, img_size, img_size, context_ratio))
    mask_in, mask_object_in, mask_context_in = get_masked_image( \
        output_dict['label'], input_bbox, bbox_cls)
    mask_out, mask_object_out, _ = get_masked_image( \
        output_dict['label'], output_bbox)

    output_dict['mask_ctx_in'] = mask_context_in.unsqueeze(0) # (1xCxHxW)
    output_dict['mask_in'] = mask_in.unsqueeze(0) # (1x1xHxW)
    output_dict['mask_out'] = mask_out.unsqueeze(0) # (1x1xHxW)
    output_dict['crop_pos'] = torch.from_numpy(crop_pos) # (1x4)
    output_dict['label'] = output_dict['label'].unsqueeze(0) # (1x1xHxW)
    output_dict['cls'] = torch.LongTensor([bbox_cls])
    if transform_img:
        output_dict['image'] = output_dict['image'].unsqueeze(0)
    #else:

    # Crop window
    x1, y1, x2, y2 = crop_pos # coordinates of crop window
    x1 = max(0,x1); y1 = max(0,y1) # make sure in range
    width = x2 - x1 + 1; height = y2 - y1 + 1

    #
    label_crop = label_original[:,:,y1:y2+1,x1:x2+1]
    input_bbox_orig = input_bbox.astype(float)
    input_bbox_orig = [input_bbox_orig[0] / opt.fineSize * width,
                       input_bbox_orig[1] / opt.fineSize * height,
                       input_bbox_orig[2] / opt.fineSize * width,
                       input_bbox_orig[3] / opt.fineSize * height]
    input_bbox_orig = np.array(input_bbox_orig)
    output_bbox_orig = output_bbox.astype(float)
    output_bbox_orig = [output_bbox_orig[0] / opt.fineSize * width,
                        output_bbox_orig[1] / opt.fineSize * height,
                        output_bbox_orig[2] / opt.fineSize * width,
                        output_bbox_orig[3] / opt.fineSize * height]
    output_bbox_orig = np.array(output_bbox_orig)
    _, _, mask_ctx_in = get_masked_image(label_crop[0], input_bbox_orig, bbox_cls)
    mask_out, _, _ = get_masked_image(label_crop[0], output_bbox_orig)

    output_dict['label_orig'] = label_crop
    output_dict['mask_ctx_in_orig'] = mask_ctx_in.unsqueeze(0)
    output_dict['mask_out_orig'] = mask_out.unsqueeze(0)

    output_bbox_global = [x1 + output_bbox_orig[0],
                          y1 + output_bbox_orig[1],
                          x1 + output_bbox_orig[2],
                          y1 + output_bbox_orig[3]]
    output_bbox_global = np.array(output_bbox_global)
    output_dict['output_bbox'] = torch.from_numpy(output_bbox)
    output_dict['output_bbox_global'] = torch.from_numpy(output_bbox_global)

    return output_dict

def paste_canvas(original, cropped, info_dict, method=Image.NEAREST,
                 resize=True, is_img=False):
    if not is_img:
        # Crop window
        x1, y1, x2, y2 = info_dict['crop_pos'].int() # coordinates of crop window
        x1 = max(0,x1); y1 = max(0,y1) # make sure in range
        x2 = min(2047,x2); y2 = min(1023,y2)

        recon = cropped[0].type('torch.FloatTensor')
    else:
        # Crop window
        x1, y1, x2, y2 = info_dict['output_bbox_global'].int() # coordinates of crop window
        x1 = max(0,x1); y1 = max(0,y1) # make sure in range
        x2 = min(2047,x2); y2 = min(1023,y2)
        width = x2 - x1 + 1; height = y2 - y1 + 1

        x3, y3, x4, y4 = info_dict['output_bbox'].int() # coordinates of crop window
        x3 = max(0,x3); y3 = max(0,y3) # make sure in range

        recon = cropped[0,:,y3:y4+1,x3:x4+1].type('torch.FloatTensor')
        recon = tensor2pil(recon, is_img) # convert to PIL image
        recon = recon.resize([width, height], method) # resize
        recon = pil2tensor(recon, is_img) # convert back to Tensor

    raw = original.clone()
    raw[0,:,y1:y2+1,x1:x2+1] = recon[:,:,:]

    return raw

