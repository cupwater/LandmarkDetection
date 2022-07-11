'''
Author: Peng Bo
Date: 2022-07-10 13:14:46
LastEditTime: 2022-07-10 13:29:51
Description: 

'''
# coding: utf8

import numpy as np
import os
import sys

IMG_SIZE = (512, 512)

if __name__ == "__main__":
    lms_path = sys.argv[1]
    lms_list = []
    with open(lms_path) as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            lms_list.append([float(l)/IMG_SIZE[0]
                            for l in line.strip().split(' ')])
    lms_list = np.array(lms_list).reshape(len(lms_list), -1, 2)

    lms_list_1 = lms_list[:, 0:14, :]
    lms_list_2 = lms_list[:, 14:20, :]
    lms_list_3 = 0.5*np.ones((lms_list.shape[0], 6, 2))

    new_lms_list = np.concatenate([lms_list_1, lms_list_3, lms_list_2], axis=1)
    new_lms_list = new_lms_list.reshape(new_lms_list.shape[0], -1)

    save_path = os.path.join(os.path.dirname(lms_path), os.path.basename(
        lms_path).split('.')[0] + '_expand.txt')
    np.savetxt(save_path, new_lms_list, fmt='%.3f')
    with open(save_path, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(str(new_lms_list.shape[0]) + '\n' + content)
