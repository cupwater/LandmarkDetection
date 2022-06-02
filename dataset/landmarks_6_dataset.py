'''
Author: Peng Bo
Date: 2022-05-14
Description: Landmarks Detection dataset

'''
import os
import time

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
import pdb

from .util import gaussianHeatmap
import matplotlib.pyplot as plt

__all__ = ['Chest6LandmarkDataset']

class Chest6LandmarkDataset(Dataset):

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
        self.idx_of_6lms = [2, 3, 4, 5, 6, 7]
        return img_data_list

    def __getitem__(self, index):
        img = self.img_data_list[index]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        w, h, _ = img.shape
        lms_img = img.copy()
        img = img.transpose((2, 0, 1))

        # get the coordinates of landmark in image
        lms = self.lms_list[index]
        # lms = [ (int(lms[i]*w), int(lms[i+1]*h)) for i in range(0, int(len(lms)), 2)]
        lms = [ (int(lms[i+1]*h), int(lms[i]*w)) for i in range(0, int(len(lms)), 2)]
        lms = [ lms[idx] for idx in self.idx_of_6lms]

        lms_heatmap = [self.genHeatmap(point, (w, h)) for point in lms]
        lms_heatmap = np.array(lms_heatmap)

        ### 将 landmarks 标记到图像 pic 上，并输出每个 landmark 的 heatmap
        if not os.path.exists('./runs/gaussianHeatmap'):
            os.makedirs('./runs/gaussianHeatmap')
        exe = str(time.time())[-6:]

        from dataset.util import getPointsFromHeatmap
        points = getPointsFromHeatmap(lms_heatmap)
        # for i,num,lm in zip(lms_heatmap, range(int(len(lms_heatmap))), lms):
        #     # 生成的 lms_heatmap 中，坐标x，y对换了
        #     plt.imsave('./runs/gaussianHeatmap/{}_{}_{}_{}.png'.format(str(exe),num,lm[0],lm[1]), i)

        # for p in points:
        #     # 回归出的坐标是正确的
        #     cv2.circle(lms_img, (int(p[1]),int(p[0])), 2, (0,255,0), 2)
        # plt.imsave('./runs/gaussianHeatmap/{}.png'.format(str(exe)), lms_img)
        ###

        img, lms_heatmap = self.transform(img, lms_heatmap)
        img, lms_heatmap = torch.FloatTensor(img), torch.FloatTensor(lms_heatmap)

        # img 正，lms_heatmap 正
        return img, lms_heatmap

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    prefix = 'data/26_landmarks'
    img_list = 'data/imglist_filter_train.txt'
    meta = 'data/lms_filter_train.txt'

    import sys
    sys.path.insert(1, 'augmentation')
    from medical_augment import LmsDetectTrainTransform
    transform = LmsDetectTrainTransform()
    chest_dataset = ChestLandmarkDataset(img_list, meta, transform, prefix)


    for i in range(chest_dataset.__len__()):
        print(i)
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
        key=cv2.waitKey(-1)
        if key != 27:
            continue
