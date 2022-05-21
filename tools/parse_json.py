'''
Author: Peng Bo
Date: 2022-05-21 11:14:06
LastEditTime: 2022-05-21 19:29:38
Description: 

'''
import json
import os
import numpy as np
import sys
import random
import pdb
import cv2

def parse_json(json_path, labels_set):
    '''
        need to process incomplete labels
    '''
    with open(json_path, 'r', encoding='utf-8') as fin:
        json_dict = json.load(fin)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']
        shapes_dict = {}
        for pos in json_dict['shapes']:
            shapes_dict[pos['label']] = [float(pos['points'][0][0])/w, float(pos['points'][0][1])/h]
        # filling the empty label with -1
        pos2d_list = [ shapes_dict[label] if label in shapes_dict else [-1, -1] for label in labels_set]

        # if len(shapes_dict) < 26:
        #     print(json_path)
        #     img = cv2.imread(json_path.replace('json', 'png'))
        #     for pos in pos2d_list:
        #         cv2.circle(img, (int(pos[0]*w), int(pos[1]*h)), 1, (255,0,0), 2)     
        #     cv2.imshow('img', img)
        #     print(pos2d_list)
        #     key = cv2.waitKey(-1)
        #     if key == 27:
        #         exit()

    return np.round(np.array(pos2d_list), 3).reshape(-1).tolist()

def main():
    img_list = sys.argv[1]
    prefix    = sys.argv[2]
    
    # acquire all the label_name
    labels_set = set()
    with open(img_list) as fin:
        for img_path in fin.readlines():
            json_path = os.path.join(prefix, img_path.strip().replace('png', 'json'))
            if os.path.exists(json_path):
                json_dict = json.load(open(json_path, 'r', encoding='utf-8'))
                labels_list = [p['label'] for p in json_dict['shapes']]
                labels_set = labels_set.union(set(labels_list))
    labels_set = sorted(labels_set)
    with open(img_list.split('.')[0] + '_labels.txt', 'w') as fout:
        fout.writelines("\n".join(list(labels_set)))

    
    lms_array_filter = []
    img_list_filter = []
    lms_array_withmask = []
    img_list_withmask = []
    with open(img_list) as fin:
        
        filter_test_idxs = []
        withmask_test_idxs = []
        ratio = 0.2
        filter_idx = 0
        withmask_idx = 0

        for img_path in fin.readlines():
            json_path = os.path.join(prefix, img_path.strip().replace('png', 'json'))
            if os.path.exists(json_path):
                pos2d_list = parse_json(json_path, labels_set)
                lms_array_withmask += pos2d_list
                img_list_withmask.append(img_path)

                if -1 not in pos2d_list:
                    lms_array_filter += pos2d_list
                    img_list_filter.append(img_path)
    
                    if random.random() < ratio:
                        filter_test_idxs.append(filter_idx)
                        withmask_test_idxs.append(withmask_idx)
                    filter_idx += 1
                withmask_idx += 1
        
        filter_all_idxs = np.array(range(len(img_list_filter)), dtype=np.int32).tolist()
        filter_train_idxs = list(set(filter_all_idxs) - set(filter_test_idxs))
        filter_train_idxs.sort()

        withmask_all_idxs = np.array(range(len(img_list_withmask)), dtype=np.int32).tolist()
        withmask_train_idxs = list(set(withmask_all_idxs) - set(withmask_test_idxs))
        withmask_train_idxs.sort()
    
    # write the image list and landmarks with mask into file
    imglist_withmask = img_list.split('.')[0] + '_withmask.txt'
    with open(imglist_withmask, 'w') as fout:
        fout.writelines("".join(img_list_withmask))
    with open(imglist_withmask.split('.')[0] + '_train.txt', 'w') as fout:
        train_imglist_withmask = [img_list_withmask[idx] for idx in withmask_train_idxs]
        fout.writelines("".join(train_imglist_withmask))
    with open(imglist_withmask.split('.')[0] + '_test.txt', 'w') as fout:
        test_imglist_withmask = [img_list_withmask[idx] for idx in withmask_test_idxs]
        fout.writelines("".join(test_imglist_withmask))
    
    # write the image list and landmarks with mask into file
    imglist_filter = img_list.split('.')[0] + '_filter.txt'
    with open(imglist_filter, 'w') as fout:
        fout.writelines("".join(img_list_filter))
    with open(imglist_filter.split('.')[0] + '_train.txt', 'w') as fout:
        train_imglist_filter = [img_list_filter[idx] for idx in filter_train_idxs]
        fout.writelines("".join(train_imglist_filter))
    with open(imglist_filter.split('.')[0] + '_test.txt', 'w') as fout:
        test_imglist_filter = [img_list_filter[idx] for idx in filter_test_idxs]
        fout.writelines("".join(test_imglist_filter))
    
    landmarks_path = os.path.join(os.path.dirname(img_list), 'lms_withmask.txt')
    lms_array_withmask = np.array(lms_array_withmask).reshape(-1, len(labels_set)*2)
    lms_array_withmask_train = lms_array_withmask[withmask_train_idxs, :]
    lms_array_withmask_test = lms_array_withmask[withmask_test_idxs, :]
    np.savetxt(landmarks_path, lms_array_withmask, fmt='%.3f')
    np.savetxt(landmarks_path.split('.')[0] + '_train.txt', lms_array_withmask_train, fmt='%.3f')
    np.savetxt(landmarks_path.split('.')[0] + '_test.txt', lms_array_withmask_test, fmt='%.3f')

    landmarks_path = os.path.join(os.path.dirname(img_list), 'lms_filter.txt')
    lms_array_filter = np.array(lms_array_filter).reshape(-1, len(labels_set)*2)
    lms_array_filter_train = lms_array_filter[filter_train_idxs, :]
    lms_array_filter_test = lms_array_filter[filter_test_idxs, :]
    np.savetxt(landmarks_path, lms_array_filter, fmt='%.3f')
    np.savetxt(landmarks_path.split('.')[0] + '_train.txt', lms_array_filter_train, fmt='%.3f')
    np.savetxt(landmarks_path.split('.')[0] + '_test.txt', lms_array_filter_test, fmt='%.3f')
    
if __name__ == '__main__':
    main()