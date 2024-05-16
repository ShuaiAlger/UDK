

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




