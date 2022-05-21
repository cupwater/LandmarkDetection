'''
Author: Peng Bo
Date: 2022-05-14
Description: Landmarks Detection dataset with mask

'''
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import pdb

from .util import gaussianHeatmap

__all__ = ['ChestLandmarkMaskDataset']

class ChestLandmarkMaskDataset(Dataset):
    '''
        since there are several landmarks without label, we generate the corresponding heatmap only for those with labels, and return the mask for computing the loss
    '''
    def __init__(self, img_list, meta, transform, prefix='data/', img_size=(512, 512), sigma=5):

        self.img_size = tuple(img_size)
        self.transform = transform
        self.prefix = prefix
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.img_data_list = self.__readAllData__()

        self.lms_list = [[float(i) for i in v.strip().split(' ')]
                      for v in open(meta).readlines()[1:]]
        
        self.genHeatmap = gaussianHeatmap(sigma, dim=len(img_size))

    def __readAllData__(self):
        img_data_list = []
        for index in range(len(self.img_list)):
            img_path = os.path.join(self.prefix, self.img_list[index].strip())
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
            img_data_list.append(img)
        return img_data_list

    def __getitem__(self, index):
        img = self.img_data_list[index]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        w, h, _ = img.shape
        img = img.transpose((2, 0, 1))
        
        # get the coordinates of landmark in image 
        lms = self.lms_list[index]
        lms = [ (int(lms[i]*w), int(lms[i+1]*h)) for i in range(int(len(lms)/2)) ]
        lms_heatmap = [self.genHeatmap(point, (w, h)) for point in lms]
        lms_heatmap = np.array(lms_heatmap)

        heatmap_mask = [1 if point[0]>0 else -1 for point in lms]

        img, lms_heatmap = self.transform(img, lms_heatmap)
        img, lms_heatmap = torch.FloatTensor(img), torch.FloatTensor(lms_heatmap)

        return img, lms_heatmap, heatmap_mask

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    prefix = 'data/landmarks/imgs'
    img_list = 'data/landmarks/img_list.txt'
    meta = 'data/landmarks/landmarks.txt'

    import sys
    sys.path.insert(1, 'augmentation')
    from medical_augment import LmsDetectTrainTransform
    transform = LmsDetectTrainTransform()
    chest_dataset = ChestLandmarkMaskDataset(img_list, meta, transform, prefix)


    for i in range(chest_dataset.__len__()):

        image, lms_heatmap = chest_dataset.__getitem__(i)
        image, lms_heatmap = image.numpy(), lms_heatmap.numpy()

        image = np.transpose(image, (1,2,0))
        image = image.astype(np.uint8)

        lms_heatmap = np.sum(lms_heatmap, axis=0) * 255
        lms_heatmap[lms_heatmap>255] = 255
        lms_heatmap = lms_heatmap.astype(np.uint8)

        # pdb.set_trace()
        cv2.imshow('heatmap', lms_heatmap)
        cv2.imshow('image', image)
        key=cv2.waitKey(500)