### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from data.segmentation_dataset import SegmentationDataset


class ADE20KDataset(SegmentationDataset):
    def initialize(self, opt):
        super(ADE20KDataset, self).initialize(opt)
        self.class_of_interest =  [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, \
            33, 35, 36, 37, 38]

    def name(self):
        return 'ADE20KDataset'
