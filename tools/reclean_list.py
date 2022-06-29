'''
Author: Peng Bo
Date: 2022-06-29 08:49:31
LastEditTime: 2022-06-29 13:57:25
Description: 

'''
# coding: utf8

import os
import sys


if __name__ == "__main__":
    img_path = sys.argv[1]
    lms_path = sys.argv[2]
    prefix   = sys.argv[3]
    
    img_list = open(img_path).readlines()
    img_list = [line.strip() for line in img_list]
    lms_list = open(lms_path).readlines()[1:]

    clean_img_list = []
    clean_lms_list = []

    for imgpath, lms in zip(img_list, lms_list):
        if os.path.exists(os.path.join(prefix, imgpath)):
            clean_img_list.append(imgpath)
            clean_lms_list.append(lms.strip())

    save_imglist_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path).split('.')[0] + '_clean.txt')
    with open(save_imglist_path, 'w') as f:
        f.write("\n".join(clean_img_list))
    save_lmslist_path = os.path.join(os.path.dirname(lms_path), os.path.basename(lms_path).split('.')[0] + '_clean.txt')
    with open(save_lmslist_path, 'w') as f:
        clean_lms_list = [str(len(clean_lms_list))] + clean_lms_list
        f.write("\n".join(clean_lms_list))

    