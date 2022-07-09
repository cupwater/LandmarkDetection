'''
Author: Peng Bo
Date: 2022-04-24 19:21:59
LastEditTime: 2022-07-09 17:30:27
Description: 

'''
import openpyxl
import numpy as np
import pdb

IMG_SIZE = (512, 512)

__all__ = ['parse_xlsx_fun', 'merge_lms_metas']


def parse_xlsx_fun(xlsx_path):
    '''
        parse the xlsx file, get the img_list and metas
    '''
    workbook = openpyxl.load_workbook(xlsx_path)
    worksheet = workbook.worksheets[0]
    img_list = []
    meta_list = []
    for row in worksheet.rows:

        meta = []
        for cell in row[-20:]:
            value = cell.value if cell.value is not None else -1
            meta.append(value)
        if -1 in meta:
            continue
        meta_list.append(meta)
        img_list.append(row[0].value.strip() + '.png')
    meta_list = meta_list[1:]
    img_list = img_list[1:]
    scores_matrix = np.array(meta_list, dtype=np.int32)
    pos_mat = 1*(scores_matrix == 0)
    # pos_mat = 1*(scores_matrix == 1)
    pos_neg_ratio = np.round(1.0*np.sum(pos_mat, axis=0) / pos_mat.shape[0], 3)
    return pos_neg_ratio, img_list, meta_list


def merge_lms_metas(imglist_path, lms_path, xlsx_path='./data/callout_summary.xlsx'):
    imglist = open(imglist_path).readlines()
    imglist = [line.strip() for line in imglist]
    lms_list = []
    with open(lms_path) as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            lms_list.append([IMG_SIZE[0]*float(l)
                            for l in line.strip().split(' ')])
    lms_list = np.array(lms_list).reshape(len(lms_list), -1, 2).tolist()

    _, whole_img_list, whole_meta_list = parse_xlsx_fun(xlsx_path)
    whole_img_list = [line.strip() for line in whole_img_list]

    filter_metas = []
    filter_imglist = []
    filter_lms = []

    nofilter_metas = []
    nofilter_imglist = []
    nofilter_lms = []

    for imgname_path, lms in zip(imglist, lms_list):
        valid_is_lms = [1 if p[0] > 0 and p[1] > 0 else 0 for p in lms]
        valid_is_lms = set(valid_is_lms)
        imgname = imgname_path.split('/')[-1]
        if imgname in whole_img_list:
            idx = whole_img_list.index(imgname)
            if 0 not in valid_is_lms:
                filter_metas.append(whole_meta_list[idx])
                filter_imglist.append(imgname_path)
                filter_lms.append(lms)
            nofilter_metas.append(whole_meta_list[idx])
            nofilter_imglist.append(imgname_path)
            nofilter_lms.append(lms)

    return filter_imglist, filter_metas, filter_lms, \
        nofilter_imglist, nofilter_metas, nofilter_lms
