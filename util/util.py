from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import re

try:
    from StringIO import StringIO
except:
    from cStringIO import StringIO
import shutil
import numpy as np


def load_image(image_file, out_size=None):
  inp_array = Image.open(image_file)
  if out_size is not None:
    inp_array = inp_array.resize(out_size)
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  return inp_array
  
def save_image(inp_array, image_file):
  """Function that dumps the image to disk."""
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  image = Image.fromarray(inp_array)
  buf = StringIO()
  image.save(buf, format='JPEG')
  with open(image_file, 'w') as f:
    f.write(buf.getvalue())

def force_mkdir(dir_name):
  if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

def force_rmdir(dir_name):
  if os.path.isdir(dir_name):
    shutil.rmtree(dir_name, ignore_errors=True)

def force_rmfile(file_name):
  if os.path.exists(file_name):
    os.remove(file_name)

def load_script_to_opt(script_path, opt_class):
    dummy_opt = opt_class().parse(save=False, default_args='')
    with open(script_path, 'r') as f:
        lines = f.readlines()
        if len(lines)==1:
            options = lines[0].split(' ')
            options = options[2:]
            options = [option.strip('\n') for option in options]
        else:
            options = []
            for line in lines:
                line = re.sub('["\']', '', line)
                option = line.split(' ')
                cmd = option[0].strip('--')
                if hasattr(dummy_opt, cmd):
                    options += option[:-1]
    
    opt = opt_class().parse(save=False, default_args=options)
    return opt

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    if image_tensor.size(0) == 1:
        image_tensor = image_tensor.repeat(3,1,1)
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2seglabel(label_tensor, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

# Converts a label map into a PIL Image
def tensor2pil(tensor, is_img=False):
    trans_fn = transforms.Compose([transforms.ToPILImage()])
    if not is_img:
        return trans_fn(tensor / 255.0)
    else:
        return trans_fn(tensor)

# Converts a PIL Image into a label map
def pil2tensor(pil_image, is_img=False):
    trans_fn = transforms.Compose([transforms.ToTensor()])
    if not is_img:
        return trans_fn(pil_image) * 255.0
    else:
        return trans_fn(pil_image)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35 or N == 36: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142),
                     (255,255,255)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
