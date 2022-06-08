'''
Author: Peng Bo
Date: 2022-05-14
Description: Landmarks Detection dataset

'''


import os
import numpy as np
import cv2
import math         


import torch
from torch.utils.data import Dataset

from .util import gaussianHeatmap, rotate, translate

__all__ = ['Chest6LandmarkDataset']


class Chest6LandmarkDataset(Dataset):

    def __init__(self, img_list, meta, transform_paras, prefix='data/', img_size=(512, 512), sigma=5):

        self.img_size = tuple(img_size)
        self.transform_paras = transform_paras
        self.prefix = prefix
        # read img_list and metas
        self.img_list = [l.strip() for l in open(img_list).readlines()]
        self.img_data_list = self.__readAllData__()
        self.idx_of_6lms = [2, 3, 4, 5, 6, 7]
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
        origin_img = self.img_data_list[index]
        img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2RGB)
        w, h, _ = img.shape
        img = img.transpose((2, 0, 1))
        # get the heatmap of landmark in image
        lms = self.lms_list[index]
        lms = [ (int(lms[i]*w), int(lms[i+1]*h)) for i in range(0, int(len(lms)), 2)]
        lms = [lms[idx] for idx in self.idx_of_6lms]
        
        lms_heatmap = [self.genHeatmap((y,x), (w, h)) for (x,y) in lms]
        lms_heatmap = np.array(lms_heatmap)

        # transform use rotate and translate
        angle = (np.random.rand()-0.5)*self.transform_paras['rotate_angle']
        offset_x = int((np.random.rand()-0.5)*self.transform_paras['offset'][0])
        offset_y = int((np.random.rand()-0.5)*self.transform_paras['offset'][1])
        
        # get rotate positions
        def lms_rotate(points_list, center, angle):
            """
            Rotate points counterclockwise by a given angle around the center.
            The angle should be given in radians.
            """
            ox, oy = center
            angle =  -np.deg2rad(angle)
            x_arr = np.array([p[0] for p in points_list])
            y_arr = np.array([p[1] for p in points_list])
            rotate_x_arr = ox + math.cos(angle) * (x_arr - ox) - math.sin(angle) * (y_arr - oy)
            rotate_y_arr = oy + math.sin(angle) * (x_arr - ox) + math.cos(angle) * (y_arr - oy)
            rotate_pos = [(int(x), int(y)) for x,y in zip(rotate_x_arr, rotate_y_arr)]
            return rotate_pos

        # rotate
        if np.random.rand() < 0.5:
            img = rotate(img, angle)
            lms_heatmap = rotate(lms_heatmap, angle)
        
        # mask those
        if np.random.rand() < 0.5:
            img = translate(img, [offset_x, offset_y])
            lms_heatmap = translate(lms_heatmap, [offset_x, offset_y])

        return torch.FloatTensor(img), torch.FloatTensor(lms_heatmap)

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    prefix = '../data/26_landmarks'
    img_list = '../data/imglist_filter_train.txt'
    meta = '../data/lms_filter_train.txt'

    transform_list = {'rotate_angle': 10, 'offset': [100,100]}
    chest_dataset = ChestLandmarkDataset(img_list, meta, transform_list, prefix)

    for i in range(chest_dataset.__len__()):
        image, lms_heatmap = chest_dataset.__getitem__(i)

        image, lms_heatmap = image.numpy(), lms_heatmap.numpy()
        print(f'max: {np.max(image)}, min:{np.min(image)}')
        image = np.transpose(image, (1,2,0)) * 255
        image = image.astype(np.uint8)
        image = cv2.merge([image[:,:,0],image[:,:,0],image[:,:,0]])

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
