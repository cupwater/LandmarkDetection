# -*- coding: utf-8 -*-
#!/bin/bash
'''
Author: Peng Bo
Date: 2022-06-27 22:19:33
LastEditTime: 2022-07-06 12:24:01
Description: Use the landmarks to judge AI quality tasks

'''

import sys
import os
import pdb
import numpy as np
import math
from sklearn.metrics import roc_curve

from tools.parse_xlsx import merge_lms_metas


semantic2idx = {
    'Clavicles_L':              0,
    'Clavicles_R':              1,
    'Costophrenic_angle_L1':    2,
    'Costophrenic_angle_L2':    3,
    'Costophrenic_angle_R1':    4,
    'Costophrenic_angle_R2':    5,
    'Lungapex_L':               6,
    'Lungapex_R':               7,
    'RIb_10th_L':               8,
    'RIb_10th_R':               9,
    'Rib_7th_L1':               10,
    'Rib_7th_L2':               11,
    'Rib_7th_R1':               12,
    'Rib_7th_R2':               13,
    'Scapulae_L1':              14,
    'Scapulae_L2':              15,
    'Scapulae_L3':              16,
    'Scapulae_R1':              17,
    'Scapulae_R2':              18,
    'Scapulae_R3':              19,
    'Spine_lowest':             20,
    'Sternoclavicular_joint_L': 21,
    'Sternoclavicular_joint_R': 22,
    'Thoracic_1th':             23,
    'Thoracic_7th':             24,
    'Trachea':                  25
}
IMG_SIZE = (512, 512)


def _symmetry_by_3points_(tgt_3pts, threshold=5):
    a = [tgt_3pts[0][0] - tgt_3pts[1][0], tgt_3pts[0][1] - tgt_3pts[1][1]]
    a = math.sqrt(a[0]*a[0] + a[1]*a[1])
    b = [tgt_3pts[0][0] - tgt_3pts[2][0], tgt_3pts[0][1] - tgt_3pts[2][1]]
    b = math.sqrt(b[0]*b[0] + b[1]*b[1])
    c = [tgt_3pts[1][0] - tgt_3pts[2][0], tgt_3pts[1][1] - tgt_3pts[2][1]]
    c = math.sqrt(c[0]*c[0] + c[1]*c[1])

    alpha = math.acos((b*b + c*c - a*a) / (2*b*c)) * 180 / math.pi
    beta = math.acos((a*a + c*c - b*b) / (2*a*c)) * 180 / math.pi
    return abs(alpha - beta)


def trachea_is_center(lms, threshold=5):
    '''
        use {Trachea, Clavicles_L, Clavicles_R} landmarks to judge whether trachea locate the center position
        return True if yes, False for no 
    '''
    Trachea_lms = lms[semantic2idx['Trachea']]
    Clavicles_L_lms = lms[semantic2idx['Clavicles_L']]
    Clavicles_R_lms = lms[semantic2idx['Clavicles_R']]
    diff = _symmetry_by_3points_(
        [Trachea_lms, Clavicles_L_lms, Clavicles_R_lms], threshold)
    return diff < threshold, diff


def thorax_is_symmetry(lms, threshold=0):
    '''
        use {Rib_7th_L2, Rib_7th_R2, Rib_7th_L1, Rib_7th_R1, Thoracic_1th} landmarks to judge whether thorax is symmetry.
        return True if yes, False for no 
    '''
    Thoracic_1th_lms = lms[semantic2idx['Thoracic_1th']]
    Rib_7th_L1_lms = lms[semantic2idx['Rib_7th_L1']]
    Rib_7th_R1_lms = lms[semantic2idx['Rib_7th_R1']]
    Rib_7th_L2_lms = lms[semantic2idx['Rib_7th_L2']]
    Rib_7th_R2_lms = lms[semantic2idx['Rib_7th_R2']]

    Rib_7th1_res = _symmetry_by_3points_(
        [Thoracic_1th_lms, Rib_7th_L1_lms, Rib_7th_R1_lms], threshold)
    Rib_7th2_res = _symmetry_by_3points_(
        [Thoracic_1th_lms, Rib_7th_L2_lms, Rib_7th_R2_lms], threshold)

    diff = Rib_7th1_res + Rib_7th2_res
    return diff < threshold, diff


def diaphragm_under_rib10th(lms, threshold=5):
    '''
        use {Costophrenic_angle_L2, Costophrenic_angle_R2, RIb_10th_L, RIb_10th_R} landmarks to judge whether diaphragm is under the 10th rib.
        return True if yes, False for no 
    '''
    Costophrenic_height = (lms[semantic2idx['Costophrenic_angle_L2']]
                           [1] + lms[semantic2idx['Costophrenic_angle_R2']][1]) / 2
    
    RIb_10th_height = (lms[semantic2idx['RIb_10th_L']]
                       [1] + lms[semantic2idx['RIb_10th_R']][1]) / 2

    diff = Costophrenic_height - RIb_10th_height
    return diff > threshold, diff


def clavicles_is_isometry(lms, threshold=5):
    '''
        use {Clavicles_L, Clavicles_R} landmarks to judge whether clavicles is isometry.
        return True if yes, False for no 
    '''
    Clavicles_L_height = lms[semantic2idx['Clavicles_L']][1]
    Clavicles_R_height = lms[semantic2idx['Clavicles_R']][1]
    diff = abs(Clavicles_L_height - Clavicles_R_height)
    return diff < threshold, diff


def Sternoclavicular_is_symmetry(lms, threshold=5):
    '''
        use {Sternoclavicular_joint_L, Sternoclavicular_joint_R, Thoracic_1th} landmarks to judge whether Sternoclaviculars is symmetry
        return True if yes, False for no 
    '''
    Thoracic_1th_lms = lms[semantic2idx['Thoracic_1th']]
    Sternoclavicular_L_lms = lms[semantic2idx['Sternoclavicular_joint_L']]
    Sternoclavicular_R_lms = lms[semantic2idx['Sternoclavicular_joint_R']]
    diff = _symmetry_by_3points_([Thoracic_1th_lms, Sternoclavicular_L_lms,
                                  Sternoclavicular_R_lms], threshold)
    return diff < threshold, diff


def thoracic_7th_is_center(lms, threshold=5):
    '''
        use Thoracic_7th to judge whether Thoracic_7th locate the center position of the chest image
        return True if yes, False for no 
    '''
    Thoracic_7th_lms = lms[semantic2idx['Thoracic_7th']]
    left_diff = abs(2*Thoracic_7th_lms[0] - IMG_SIZE[0])
    right_diff = abs(2*Thoracic_7th_lms[1] - IMG_SIZE[1])
    diff = 0.5 * (left_diff + right_diff)
    return diff < threshold, diff


def spine_parallel_vertical(lms, threshold=3):
    '''
        use {Thoracic_1th, Spine_lowest} to judge whether spine is parallel along with the vertical axis of the chest image
        return True if yes, False for no 
    '''
    thoracic_1th_lms = lms[semantic2idx['Thoracic_1th']]
    spine_lowest_lms = lms[semantic2idx['Spine_lowest']]
    spine_vec = [thoracic_1th_lms[0] - spine_lowest_lms[0], thoracic_1th_lms[1] - spine_lowest_lms[1]]
    angle_yaxis = math.atan(spine_vec[0] / spine_vec[1])

    return angle_yaxis < threshold, angle_yaxis


def thoracic1th_35cm2upper(lms, threshold=1, cm_per_pixel=1):
    '''
        use Thoracic_1th to judge whether the distance between Thoracic_1th and upper is about 3~5cm.
        return True if yes, False for no 
    '''
    thoracic_1th_lms = lms[semantic2idx['Thoracic_1th']]
    diff = thoracic_1th_lms[1] * cm_per_pixel - 4
    return diff < threshold, diff


def thorax_5cm2edge(lms, threshold=1, cm_per_pixe=1):
    '''
        use {Costophrenic_angle_L2, Costophrenic_angle_R2} to judge whether the distance between Costophrenic_angles and edge(left and right) is small than 5cm.
        return True if yes, False for no 
    '''
    Costophrenic_angle_L2_lms = lms[semantic2idx['Costophrenic_angle_L2']]
    Costophrenic_angle_R2_lms = lms[semantic2idx['Costophrenic_angle_R2']]
    # left and right distance
    ldis = Costophrenic_angle_L2_lms[0] * cm_per_pixe
    rdis = (IMG_SIZE[0] - Costophrenic_angle_R2_lms[0]) * cm_per_pixe

    diff = abs(ldis - 5) + abs(rdis - 5)
    cond = Costophrenic_angle_L2_lms[0] > 0 and Costophrenic_angle_R2_lms[0] < IMG_SIZE[0]
    return diff < threshold and cond, diff


def lms2predict(lms):
    '''
        use landmarks to judge several AI medical image quality tasks 
        return the judge result
    '''
    return True


RULES_LIST = [
    trachea_is_center,
    thorax_is_symmetry,
    diaphragm_under_rib10th,
    clavicles_is_isometry,
    Sternoclavicular_is_symmetry,
    thoracic_7th_is_center,
    spine_parallel_vertical,
    thoracic1th_35cm2upper,
    thorax_5cm2edge
]
RULES_IDX = [idx-20 for idx in [23, 26, 27, 28, 29, 33, 34, 35, 36]]


def get_optimal_threshold(lms_list, gt_labels_list, task_id):
    rule_fun = RULES_LIST[task_id]
    diff_list = []
    for lms, _ in zip(lms_list, gt_labels_list):
        _, diff = rule_fun(lms)
        diff_list.append(diff)

    fpr, tpr, thresholds = roc_curve(gt_labels_list, diff_list)

    def Find_Optimal_Cutoff(TPR, FPR, threshold):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = threshold[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold, point
    optimal_threshold, _ = Find_Optimal_Cutoff(fpr, tpr, thresholds)
    return optimal_threshold, fpr, tpr, thresholds


def get_accuracy(lms_list, gt_labels_list, task_id, threshold):
    rule_fun = RULES_LIST[task_id]
    pred_list = []
    correct_num = 0
    for lms, gt in zip(lms_list, gt_labels_list):
        pred, _ = rule_fun(lms, threshold)
        pred_list.append(pred)
        if pred == gt:
            correct_num += 1
    return 1.0*correct_num / len(gt_labels_list)
    


if __name__ == "__main__":

    imglist_path = sys.argv[1]
    lms_path = sys.argv[2]

    filter_imglist, filter_metas, filter_lms = merge_lms_metas(
        imglist_path, lms_path)
    optimal_threshold_list = []

    accuracies = []
    for task_idx in range(len(RULES_LIST)):
        gt_labels = [int(metas[RULES_IDX[task_idx]]) for metas in filter_metas]
        optimal_threshold, fpr, tpr, thresholds = get_optimal_threshold(
            filter_lms, gt_labels, 0)
        optimal_threshold_list.append(optimal_threshold)
        acc = get_accuracy(filter_lms, gt_labels, task_idx, optimal_threshold)
        accuracies.append(acc)

    save_path = os.path.join(os.path.dirname(
        lms_path), 'optimal_thresholds.txt')
    np.savetxt(save_path, np.array(optimal_threshold_list), fmt='%.3f')

    save_path = os.path.join(os.path.dirname(
        lms_path), 'accuracies.txt')
    np.savetxt(save_path, np.array(accuracies), fmt='%.3f')
