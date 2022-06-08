'''
Author: Peng Bo
Date: 2022-05-14
Description: Landmarks Detection dataset

'''
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from .util import gaussianHeatmap

import sys
sys.path.insert(1, '../augmentation')
from .medical_augment import rotate, translate

__all__ = ['ChestLandmarkDataset']



class ChestLandmarkDataset(Dataset):

    def __init__(self, img_list, meta, transform_paras, prefix='data/', img_size=(512, 512), sigma=5):

        self.img_size = tuple(img_size)
        self.transform_paras = transform_paras
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
            # print(f'max: {np.max(img)}, min:{np.min(img)}')
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
        lms = [ (int(lms[i+1]*h), int(lms[i]*w)) for i in range(0, int(len(lms)), 2)]

        lms_heatmap = [self.genHeatmap(point, (w, h)) for point in lms]
        lms_heatmap = np.array(lms_heatmap)

        def trans(*imgs, transform_list=None):
            ''' img: chanel x imageshape
            '''
            ret = []
            for img in imgs:
                # copy is necessary, to avoid modifying origin data
                cur_img = img.copy()
                for f in transform_list:
                    cur_img = f(cur_img)
                    # print(f'max: {np.max(cur_img)}, min:{np.min(cur_img)}')
                # copy is necessary, torch needs ascontiguousarray
                ret.append(cur_img.copy())
            return tuple(ret)

        transform_list = []
        # transform
        if np.random.rand() < 0.5:
            transform_list.append(rotate((np.random.rand()-0.5)*self.transform_paras['rotate_angle']))
        if np.random.rand() < 0.5:
            rorate_x = int((np.random.rand()-0.5)*self.transform_paras['offset'][0])
            rorate_y = int((np.random.rand()-0.5)*self.transform_paras['offset'][1])
            transform_list.append(translate([rorate_x, rorate_y]))
    
        img, lms_heatmap = trans(img, lms_heatmap, transform_list=transform_list)
        img, lms_heatmap = torch.FloatTensor(img), torch.FloatTensor(lms_heatmap)

        return img, lms_heatmap

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    prefix = '../data/26_landmarks'
    img_list = '../data/imglist_filter_train.txt'
    meta = '../data/lms_filter_train.txt'

    transform_list = {'rotate_angle': 1, 'offset': [1,1]}
    chest_dataset = ChestLandmarkDataset(img_list, meta, transform_list, prefix)

    for i in range(chest_dataset.__len__()):
        image, lms_heatmap = chest_dataset.__getitem__(i)
        image, lms_heatmap = image.numpy(), lms_heatmap.numpy()
        print(f'max: {np.max(image)}, min:{np.min(image)}')
        image = np.transpose(image, (1,2,0)) * 255
        image = image.astype(np.uint8)

        lms_heatmap = np.sum(lms_heatmap, axis=0) * 255
        lms_heatmap[lms_heatmap>255] = 255
        lms_heatmap = lms_heatmap.astype(np.uint8)
        cv2.imshow('heatmap', lms_heatmap)
        
        cv2.imshow('image', image)
        
        key=cv2.waitKey(-1)
        if key != 27:
            continue
        else:
            exit(0)
