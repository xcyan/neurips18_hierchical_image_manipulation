"""file defining mask loss."""

import torch
import torch.nn as nn
from torch.autograd import Variable
from layer_util import *
import numpy as np
import functools

IGNORE_INDEX = 255
      
class MaskReconLoss(nn.Module):
    def __init__(self, use_nll=True): 
        super(MaskReconLoss, self).__init__()
        assert use_nll
        self.criterion = nn.NLLLoss2d(ignore_index=IGNORE_INDEX) 

    def forward(self, pred_logit, gt_label_, gt_mask):
        indices = (gt_mask < 0.5).nonzero()
        gt_label = gt_label_.clone()
        if len(indices) > 0:
          try:
            gt_label[indices[:, 0], indices[:, 2], indices[:, 3]] = IGNORE_INDEX
          except:
            import pdb; pdb.set_trace()
        loss = self.criterion(input=pred_logit, target=gt_label)
        return loss

