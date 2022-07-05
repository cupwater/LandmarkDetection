'''
Author: Peng Bo
Date: 2022-06-29 14:00:46
LastEditTime: 2022-07-06 01:34:46
Description: 

'''
# -*- coding: utf-8 -*-
#!/bin/bash
'''
Author: Peng Bo
Date: 2022-06-27 10:54:30
LastEditTime: 2022-07-05 23:21:52
Description: Align the Chest Image using landmarks according to a reference image

'''
import numpy as np
import cv2
import sys
import os
import pdb

from skimage import transform as trans

IMG_SIZE = (512, 512)
# TGT_IDXs = [2,3,4,5,6,7]
TGT_IDXs = [3,5,6,7]
pts_num = len(TGT_IDXs)

ref_points = [[75, 405], [437, 408], [198, 63], [320, 61]]

def reference_landmarks(lms_array):
    lmss = np.mean(lms_array, axis=0).reshape(-1, 2)
    avg_lms = [ [int(lmss[i][0]), int(lmss[i][1])] for i in range(lmss.shape[0]) ]
    return avg_lms


def get_affine_transform_matrix(src_pts, dst_pts):
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img, facial_pts, reference_pts, raw_lms=None, crop_size=IMG_SIZE):
    
    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    tform = trans.SimilarityTransform()
    tform.estimate(src_pts, ref_pts)
    tfm = tform.params[0:2, :]
    
    if raw_lms is not None:
        mask = [ 1 if p[0]>0 and p[1]>0 else 0 for p in raw_lms.tolist() ]
        pts = np.concatenate((raw_lms, np.ones(raw_lms.shape[0]).reshape(raw_lms.shape[0], -1)), axis=1)
        transformed_pts = np.dot(tfm, pts.T).T
        mask = np.array(mask).reshape(-1,1)
        transformed_pts = np.repeat(mask, 2, axis=1) * transformed_pts
        transformed_pts[transformed_pts==0] = -IMG_SIZE[0]
        transformed_pts[transformed_pts>IMG_SIZE[0]] = -IMG_SIZE[0]
        transformed_pts[transformed_pts<0] = -IMG_SIZE[0]
        # pdb.set_trace()
    else:
        transformed_pts = None
    
    aligned_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))
    return aligned_img, transformed_pts


if __name__ == "__main__":
    img_path = sys.argv[1]
    lms_path = sys.argv[2]
    prefix   = sys.argv[3]
    out_prefix = sys.argv[4]

    img_list = open(img_path).readlines()
    img_list = [line.strip() for line in img_list]
    lms_list = []
    with open(lms_path) as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            lms_list.append([ IMG_SIZE[0]*float(l) for l in line.strip().split(' ')])
    lms_array = np.array(lms_list).reshape(len(lms_list), -1, 2)
    # ref_points = reference_landmarks(lms_array[:, TGT_IDXs, :])

    aligned_lms_list = []
    for idx in range(len(img_list)):
        src_img = img_list[idx]
        src_lms = lms_array[idx][TGT_IDXs, :]
        raw_lms = lms_array[idx]
        src_img = cv2.imread(os.path.join(prefix, src_img))
        aligned_img, transformed_pts = warp_and_crop_face(src_img, src_lms, ref_points, raw_lms, crop_size=IMG_SIZE)

        save_path = os.path.join(out_prefix, img_list[idx])
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, aligned_img)

        aligned_lms_list.append(transformed_pts)
    aligned_lms_list = np.array(aligned_lms_list).reshape(len(img_list), -1, 2)
    # normalize to 0~1
    aligned_lms_list[:, :, 0] = aligned_lms_list[:, :, 0]/IMG_SIZE[0]
    aligned_lms_list[:, :, 1] = aligned_lms_list[:, :, 1]/IMG_SIZE[1]
    aligned_lms_list = aligned_lms_list.reshape(len(img_list), -1)

    save_path = os.path.join(out_prefix, os.path.basename(lms_path))
    np.savetxt(save_path, aligned_lms_list, fmt='%.3f')
    with open(save_path, 'r+') as f:  
        content = f.read()
        f.seek(0, 0)
        f.write(str(aligned_lms_list.shape[0]) + '\n' + content)

        # if transformed_pts is not None:
        #     for i in range(transformed_pts.shape[0]):
        #         p = (int(transformed_pts[i][0]), int(transformed_pts[i][1]))
        #         cv2.circle( aligned_img, p, 1, (255,0,0), 2 )
        # cv2.imshow('img', aligned_img)
        # key = cv2.waitKey(-1)
        # if key == 27:
        #     exit(-1)
