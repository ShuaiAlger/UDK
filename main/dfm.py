#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:09:25 2021

@author: ufukefe
"""

import os
import argparse
import yaml
import cv2
from DeepFeatureMatcher import DeepFeatureMatcher

from QDeepFeatureMatcher import QDeepFeatureMatcher
from SegDeepFeatureMatcher import SegDeepFeatureMatcher
from TransformerDeepFeatureMatcher import TRDeepFeatureMatcher



from ManyDeepFeatureMatcher import ManyDeepFeatureMatcher

# from ManyDFM import ManyDeepFeatureMatcher


import tqdm


from PIL import Image
import numpy as np
import time

#To draw_matches
def draw_matches(img_A, img_B, keypoints0, keypoints1):
    
    p1s = []
    p2s = []
    dmatches = []
    for i, (x1, y1) in enumerate(keypoints0):
         
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
        
    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s, 
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    
    return matched_images




import os
import glob
import numpy as np
import cv2


def get_H_matrix(h_file_path):
    file = open(h_file_path, 'r')
    h_split = file.read().split()
    H = np.zeros((3, 3))
    for j in range(3):
        for i in range(3):
            H[j][i] = h_split[j * 3 + i]
    return H


def get_gt_H_matrix(data_path, id1, id2):
    if id1 == 1:
        h_filename = "H_" + str(id1) + "_" + str(id2)
        return get_H_matrix(data_path + "/" + h_filename)
    else:
        h_file1 = "H_1_" + str(id1)
        h_file2 = "H_1_" + str(id2)
        H1 = get_H_matrix(data_path + "/" + h_file1)
        H2 = get_H_matrix(data_path + "/" + h_file2)
        return np.linalg.inv(H1)@H2


def get_MMA(_H_gt, _m_pts1, _m_pts2, _thres):
    t_ = _thres*_thres
    N_ = len(_m_pts1)
    sum_value_ = 0
    for i in range(N_):
        new_pt = _H_gt @ np.array([_m_pts1[i][0], _m_pts1[i][1], 1])
        new_pt /= new_pt[2]
        du = new_pt[0] - _m_pts2[i][0]
        dv = new_pt[1] - _m_pts2[i][1]
        if (du*du + dv*dv) < t_:
            sum_value_ = sum_value_ + 1
    return sum_value_/N_ if sum_value_ > 0 else 0.0




def eval_matches(p1s, p2s, homography):
    # Compute the reprojection errors from im1 to im2 
    # with the given the GT homography
    p1s_h = np.concatenate([p1s, np.ones([p1s.shape[0], 1])], axis=1)  # Homogenous
    p2s_proj_h = np.transpose(np.dot(homography, np.transpose(p1s_h)))
    p2s_proj = p2s_proj_h[:, :2] / p2s_proj_h[:, 2:]
    dist = np.sqrt(np.sum((p2s - p2s_proj) ** 2, axis=1))
    return dist





def draw_green_red(_img1, _img2, _H_gt, _m_pts1, _m_pts2, _thres, _thinkness):
    canvas = np.concatenate([_img1, _img2], axis=1)
    t_ = _thres*_thres
    N_ = len(_m_pts1)
    sum_value_ = 0
    for i in range(N_):
        new_pt = _H_gt @ np.array([_m_pts1[i][0], _m_pts1[i][1], 1])
        new_pt /= new_pt[2]
        du = new_pt[0] - _m_pts2[i][0]
        dv = new_pt[1] - _m_pts2[i][1]
        p1 = (int(_m_pts1[i][0]), int(_m_pts1[i][1]))
        p2 = (int(_m_pts2[i][0]) + _img1.shape[1], int(_m_pts2[i][1]))
        if (du*du + dv*dv) < t_:
            sum_value_ = sum_value_ + 1
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.line(canvas, p1, p2, color, _thinkness)
    return canvas


class RHpatchesDataLoader():
    def __init__(self, dataset_path, img_format, gray=True):
        self.dataset_path = dataset_path
        self.pairs = []
        dirlist = []
        self.gray = gray
        for root, dirnames, filenames in os.walk(dataset_path):
            dirnames = sorted(dirnames)
            for dirname in dirnames:
                dirlist.append(os.path.join(root, dirname))
                data_path = os.path.join(root, dirname)
            #   traverse images
                image_pairs = glob.glob(data_path + "/*" + img_format)
                # image_pairs = sorted(image_pairs) bug
                image_pairs = [data_path+f"/{i}"+img_format for i in range(1, len(image_pairs)+1)]
                image_num = len(image_pairs)
                for i in range(1, image_num):
                    id1 = 1
                    id2 = i + 1
                    real_H = get_gt_H_matrix(data_path, id1, id2)
                    self.pairs.append([image_pairs[0], image_pairs[i], id1, id2, real_H])
        self.read_idx = 0
        self.read_pair = []
        self.length = len(self.pairs)

    def get_length(self):
        return self.length

    def next_item(self):
        self.read_pair = self.pairs[self.read_idx]
        self.read_idx = self.read_idx + 1

    def read_data(self):
        self.next_item()
        img1 = cv2.imread(self.read_pair[0], 0)
        img2 = cv2.imread(self.read_pair[1], 0)
        id1 = self.read_pair[2]
        id2 = self.read_pair[3]
        real_H = self.read_pair[4]
        return img1, img2, id1, id2, real_H

    def read_data_from_index(self, index : int):
        if self.gray:
            img1 = cv2.imread(self.pairs[index][0], 0)
            img2 = cv2.imread(self.pairs[index][1], 0)
            id1 = self.pairs[index][2]
            id2 = self.pairs[index][3]
            real_H = self.pairs[index][4]
            return img1, img2, id1, id2, real_H
        else:
            img1 = cv2.imread(self.pairs[index][0])
            img2 = cv2.imread(self.pairs[index][1])
            id1 = self.pairs[index][2]
            id2 = self.pairs[index][3]
            real_H = self.pairs[index][4]
            return img1, img2, id1, id2, real_H

    def read_H_from_index(self, index : int):
        real_H = self.pairs[index][4]
        return real_H

    def get_dataset_name(self):
        return self.dataset_path.split('/')[-1]







def evaluate(rhdl, km, DRAW_IMAGE=True):
    paris_num = rhdl.get_length()
#   init the storage of results
    results = np.zeros((paris_num, 10), dtype=np.float32)
    kpts_nums = np.zeros((paris_num, 1), dtype=np.float32)
    count = 0
    start = time.time()


    for index in tqdm.tqdm(range(0, rhdl.get_length()), desc="matching"):

        img1, img2, id1, id2, real_H = rhdl.read_data_from_index(index)

        
        pointsA, pointsB = km(img1, img2)

        kpts_nums[index] = pointsA.shape[0]
        if pointsA.shape[0] < 2:
            mma = np.zeros(10)
        else:
            kpts1 = pointsA
            kpts2 = pointsB
            distances = eval_matches(kpts1, kpts2, real_H)
            if distances.shape[0] >= 1:
                mma = np.around(np.array([np.count_nonzero(distances <= i)/distances.shape[0] 
                            for i in range (1,11)]),3)
            else:
                mma = np.zeros(10)
        results[count] = mma

        count = count + 1
        


    result_str = ""
    for thres in range(1, 11):
        result_str = result_str + str(thres) + " : " + str(results[:, thres-1].mean()) + ","

    print(result_str)

    end = time.time()
    running_time = end - start




# class ManyMatcher():
#     def __init__(self, backbone):
#         self.backbone = backbone
#         with open(str("configs/"+backbone+".yml"), "r") as configfile:
#             self.config = yaml.safe_load(configfile)['configuration']
        
#         self.fm = ManyDeepFeatureMatcher(enable_two_stage = self.config['enable_two_stage'], model = self.config['model'], 
#                         ratio_th = self.config['ratio_th'], bidirectional = self.config['bidirectional'], )

#     def match(self, img_A, img_B):
#         H, H_init, points_A, points_B = self.fm.match(img_A, img_B)
#         keypoints0 = points_A.T
#         keypoints1 = points_B.T

#         return H, H_init, points_A, points_B






def demo_hpatches(function_match):

    dataset_path='/media/shuai/Correspondence/DATASETS/hpatches-sequences-release'
    img_format = ".ppm"

    rhdl = RHpatchesDataLoader(dataset_path, img_format, gray=False)

    evaluate(rhdl, function_match)


def demo_camera_pose(function_match):


# eval megadepth
    from eval_megadepth_scannet_yfcc import evaluate_megadepth

    # evaluate_megadepth(manym.match)

    # evaluate_megadepth(manym.match_rotation, USE_ROT90_DATASET=1, USE_CONTINUE_ROT_DATASET=0)

    evaluate_megadepth(function_match, USE_ROT90_DATASET=0, USE_CONTINUE_ROT_DATASET=0)




#Take arguments and configurations
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Algorithm wrapper for Image Matching Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_pairs', type=str, default='image_pairs.txt')
    args = parser.parse_args() 


    model_name = "vgg19"
    # model_name = "vgg19_bn"
    # model_name = "resnet18"


    PRINT_LAYERS = False


    with open(str("configs/" + model_name + ".yml"), "r") as configfile:
        config = yaml.safe_load(configfile)['configuration']

    if PRINT_LAYERS:
        print(len(config['layers']))
        for i in range(len(config['layers'])):
            print(i, " : ", config['layers'][i])
        exit(0)

    # Make result directory
    # os.makedirs(config['output_directory'], exist_ok=True)

    # manym = DeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
    #                     ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )

    manym = None

    if 1:
        manym = ManyDeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
                        ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], layers = config['layers'])
    else:
        manym = QDeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = config['model'], 
                            ratio_th = config['ratio_th'], bidirectional = config['bidirectional'])

    # manym = TRDeepFeatureMatcher(enable_two_stage = config['enable_two_stage'], model = 'swin_b',
    #                     ratio_th = config['ratio_th'], bidirectional = config['bidirectional'], )



    demo_camera_pose(manym.match_multiscale)






    # demo_hpatches(manym.match_multiscale)




# resnet18
# 1 : 0.46991667,2 : 0.70497775,3 : 0.80142593,4 : 0.84850925,5 : 0.86683893,6 : 0.8791611,7 : 0.8843296,8 : 0.88701665,9 : 0.88862586,10 : 0.88925374, mmn thres 0.70
# 1 : 0.4574815,2 : 0.7117704,3 : 0.8167759,4 : 0.8658037,5 : 0.8876611,6 : 0.9029629,7 : 0.9101166,8 : 0.9144537,9 : 0.91674817,10 : 0.91804254, mmn thres 0.75
# 1 : 0.4510463,2 : 0.71319634,3 : 0.82686484,4 : 0.8806722,5 : 0.9052037,6 : 0.9192352,7 : 0.9265611,8 : 0.93233335,9 : 0.935263,10 : 0.9388389, # mnn thres 0.80
# 1 : 0.4317593,2 : 0.6939315,3 : 0.8101111,4 : 0.8684019,5 : 0.89595556,6 : 0.9126759,7 : 0.92225,8 : 0.9296074,9 : 0.9341852,10 : 0.9369, # mnn thres 0.85
# 1 : 0.40723154,2 : 0.66401106,3 : 0.7817,4 : 0.8425352,5 : 0.8726407,6 : 0.8915352,7 : 0.90311855,8 : 0.9112149,9 : 0.91740924,10 : 0.9211389, # mnn thres 0.90
# 1 : 0.3753278,2 : 0.6198482,3 : 0.736387,4 : 0.7983889,5 : 0.8309074,6 : 0.8525648,7 : 0.86613333,8 : 0.876413,9 : 0.88428336,10 : 0.88951486, # mnn thres 0.95
# 1 : 0.3339,2 : 0.5581778,3 : 0.66905737,4 : 0.73042035,5 : 0.76470745,6 : 0.78823525,7 : 0.8042389,8 : 0.8167185,9 : 0.82653517,10 : 0.83360744, # mnn thres 1.0


# resnet34
# 1 : 0.3175037,2 : 0.5424888,3 : 0.66257405,4 : 0.7331389,5 : 0.77535737,6 : 0.8046945,7 : 0.8248518,8 : 0.8403666,9 : 0.8523611,10 : 0.8609593, # mnn thres 1.0
# 1 : 0.37643516,2 : 0.6338611,3 : 0.7610963,4 : 0.82965004,5 : 0.86728334,6 : 0.89197963,7 : 0.90734625,8 : 0.9185611,9 : 0.92612594,10 : 0.93103886, # mnn thres 0.95
# 1 : 0.4149,2 : 0.6761574,3 : 0.7997685,4 : 0.8625518,5 : 0.89592963,6 : 0.91506475,7 : 0.9274852,8 : 0.9363463,9 : 0.94069076,10 : 0.943574, # mnn thres 0.90
# 1 : 0.42709813,2 : 0.6768537,3 : 0.79276115,4 : 0.85131854,5 : 0.8799352,6 : 0.8943148,7 : 0.9033111,8 : 0.90959257,9 : 0.91285926,10 : 0.9151296, # mnn thres 0.85
# 1 : 0.42784446,2 : 0.6580166,3 : 0.758163,4 : 0.80618334,5 : 0.8281574,6 : 0.83809817,7 : 0.8445741,8 : 0.84838516,9 : 0.85131115,10 : 0.8530593, # mnn thres 0.80
# 1 : 0.4160815,2 : 0.6170963,3 : 0.6964667,4 : 0.7392852,5 : 0.7536611,6 : 0.7622981,7 : 0.76572037,8 : 0.7676333,9 : 0.7705945,10 : 0.77117217, # mnn thres 0.75
# 1 : 0.39284262,2 : 0.5534963,3 : 0.61611664,4 : 0.64979815,5 : 0.66150373,6 : 0.66861296,7 : 0.67107964,8 : 0.6723759,9 : 0.67348146,10 : 0.6739685, # mnn thrse 0.70


# resnet101
# 1 : 0.31662408,2 : 0.53746486,3 : 0.654363,4 : 0.72329444,5 : 0.7652981,6 : 0.79530925,7 : 0.81634444,8 : 0.83291477,9 : 0.845813,10 : 0.85536486, # mnn thres 1.0
# 1 : 0.33151484,2 : 0.56136113,3 : 0.6810759,4 : 0.75039625,5 : 0.79145557,6 : 0.82005554,7 : 0.8399778,8 : 0.855226,9 : 0.8670574,10 : 0.87562406, # mnn thres 0.99
# 1 : 0.37974074,2 : 0.63546294,3 : 0.7598111,4 : 0.82655185,5 : 0.86327595,6 : 0.88692963,7 : 0.90229255,8 : 0.9129537,9 : 0.9217093,10 : 0.9279537, # mnn thres 0.95
# 1 : 0.41659072,2 : 0.68384266,3 : 0.80372405,4 : 0.86596483,5 : 0.8972185,6 : 0.9164944,7 : 0.92867035,8 : 0.9358185,9 : 0.94221115,10 : 0.94716483, # mnn thres 0.90
# 1 : 0.43324447,2 : 0.691587,3 : 0.80431294,4 : 0.85724074,5 : 0.8833501,6 : 0.8977111,7 : 0.9073926,8 : 0.9124704,9 : 0.9156204,10 : 0.9184963, # mnn thres 0.85
# 1 : 0.42924446,2 : 0.65931296,3 : 0.75604814,4 : 0.80088896,5 : 0.8220222,6 : 0.8336556,7 : 0.8386315,8 : 0.8427611,9 : 0.8449537,10 : 0.8460241, # mnn thres 0.80
# 1 : 0.40340555,2 : 0.5983537,3 : 0.6777186,4 : 0.71415746,5 : 0.7281907,6 : 0.7378852,7 : 0.7407648,8 : 0.7447148,9 : 0.74663144,10 : 0.7473481, # mmn thres 0.75
# 1 : 0.37862593,2 : 0.5441778,3 : 0.6054037,4 : 0.63516295,5 : 0.6466408,6 : 0.65346295,7 : 0.65557224,8 : 0.6569,9 : 0.6579111,10 : 0.6582889, # mmn thres 0.70











