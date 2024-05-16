

import time
import tqdm



#   folder operation   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os

def mkdir_with_check(_dir):
    if not os.path.exists(_dir):
        os.mkdir(_dir)
        return
    else:
        return

def get_sub_dirs(_path):
    sub_dirs_ = []
    for i in os.scandir(_path):
        if i.is_dir():
            sub_dirs_.append(i.path)
    return sorted(sub_dirs_)

def get_files(_path):
    files_ = []
    for i in os.scandir(_path):
        if i.is_file():
            files_.append(i.path)
    return sorted(files_)


#   LOG   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import logging

class CustomFormatter(logging.Formatter):

    reset = "\x1b[0m"

    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    purple = "\x1b[35;20m"
    cyan = "\x1b[36;20m"
    white = "\x1b[37;20m"
    grey = "\x1b[38;20m"
    
    bold_red = "\x1b[31;1m"
    bold_blue = "\x1b[34;1m"
    bold_red_bgwhile = "\x1b[31;47;1m"

    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG:    blue     + format + reset,
        logging.INFO:     green    + format + reset,
        logging.WARNING:  yellow   + format + reset,
        logging.ERROR:    red      + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def fancy_color():
    colored = [0] + [0x5f + 40 * n for n in range(0, 5)]
    colored_palette = [
        "%02x/%02x/%02x" % (r, g, b) 
        for r in colored
        for g in colored
        for b in colored
    ]

    grayscale = [0x08 + 10 * n for n in range(0, 24)]
    grayscale_palette = [
        "%02x/%02x/%02x" % (a, a, a)
        for a in grayscale 
    ]

    normal = "\033[38;5;%sm" 
    bold = "\033[1;38;5;%sm"
    reset = "\033[0m"

    for (i, color) in enumerate(colored_palette + grayscale_palette, 16):
        index = (bold + "%4s" + reset) % (i, str(i) + ':')
        hex   = (normal + "%s" + reset) % (i, color)
        newline = '\n' if i % 6 == 3 else ''
        print(index, hex, newline, end='')

#   LOG   INSTANCE   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def create_logger(_filename='ODDM.log'):
    # create logger with 'spam_application'
    logger = logging.getLogger("ODDM")
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(_filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


#   .h5 file saver and reader of keypoints matching  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import h5py

def save_h5(_img_name, _mtchs, _filename='undefined.h5'):
    h5_imgname = str(_img_name)
    h5_matches = np.array(_mtchs)
    h5file = h5py.File(_filename, 'w')
    h5file.create_dataset('imgname', data=h5_imgname)
    h5file.create_dataset('matches', data=h5_matches)
    h5file.close()

def read_h5(_filename):

    h5file = h5py.File(_filename, 'r')
    # for k in h5_file.keys():
    #     print(k)
    h5_imgname = h5file['imgname']
    h5_matches = h5file['matches']
    return h5_imgname, h5_matches


# #   tool-function
import torch
def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


#   display data
import numpy as np
import matplotlib.pyplot as plt
def showHistogram(_matrix):
    
    histogram, bin_edges = np.histogram(_matrix, bins=60, range=(-20, 60))

    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([-20.0, 60.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()






# evaluate




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
    
    



logger = create_logger('0920_sim2e_exp.log')

def write_line_txt(data_str, filename="result.txt"):
    with open(filename, "a+") as f:
        f.write(data_str+"\n")
        f.close()



def evaluate(rhdl, km, DRAW_IMAGE=True):
    paris_num = rhdl.get_length()
#   init the storage of results
    results = np.zeros((paris_num, 10), dtype=np.float32)
    kpts_nums = np.zeros((paris_num, 1), dtype=np.float32)
    count = 0
    start = time.time()

    draw_img_save_path = "./display_1212/"
    mkdir_with_check(draw_img_save_path)

    seq = rhdl.get_dataset_name()[-5:]
    draw_img_save_path = draw_img_save_path + seq + "/"
    mkdir_with_check(draw_img_save_path)

    draw_img_save_path = draw_img_save_path + km.get_method() + "/"
    mkdir_with_check(draw_img_save_path)

    print(">>>>>>>>>>>>      ", paris_num)

    # for index in tqdm(range(0, paris_num), desc="matching"):
    # for index in tqdm(range(513, 514), desc="matching"):
    # for index in tqdm(range(82, 84), desc="matching"):
    for index in tqdm.tqdm(range(0, paris_num)):
        img1, img2, id1, id2, real_H = rhdl.read_data_from_index(index)

        kpts1, kpts2 = km.run(img1, img2)

        if kpts1.shape[0] < 2:
            mma = np.zeros(10)
        else:
            distances = eval_matches(kpts1, kpts2, real_H)
            if distances.shape[0] >= 1:
                mma = np.around(np.array([np.count_nonzero(distances <= i)/distances.shape[0] 
                            for i in range (1,11)]),3)
            else:
                mma = np.zeros(10)
        results[count] = mma

        count = count + 1
        
        # if DRAW_IMAGE:
        #     draw_img = draw_green_red(img1, img2, real_H, res[:, 0:2], res[:, 2:], 3, 2)
        #     cv2.imwrite(draw_img_save_path + km.get_method() + rhdl.get_dataset_name() + "_" +
        #                                         str(index).zfill(4) + "_" +
        #                                         ".png", draw_img)

    result_str = ""
    for thres in range(1, 11):
        result_str = result_str + str(thres) + " : " + str(results[:, thres-1].mean()) + ","
    logger.info(result_str)
    logger.info("")
    logger.info("kpts_nums: "+str(kpts_nums.mean()))

    write_line_txt(result_str)

    end = time.time()
    running_time = end - start
    logger.info("running_time: "+str(float(running_time / paris_num)))
    logger.info("")


def flann_match(_kpts1, _kpts2, _desc1, _desc2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    FLANN = cv2.FlannBasedMatcher(indexParams = index_params, searchParams = search_params)

    _desc1 = np.float32(_desc1)
    _desc2 = np.float32(_desc2)

    if _desc1.shape[0] < 4:
        return np.empty((0, 0))

    if _desc2.shape[0] < 4:
        return np.empty((0, 0))

    matches = FLANN.knnMatch(queryDescriptors = _desc1, trainDescriptors = _desc2, k = 2)
    ratio_thresh = 0.7
    good_matches = []

    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # output = cv2.drawMatches(img1 = image1, keypoints1 = _kpts1, img2 = image2, keypoints2 = _kpts2,
    #             matches1to2 = good_matches, outImg = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    results = np.zeros((len(good_matches), 4))
    for j in range(len(good_matches)):
        id2 = good_matches[j].trainIdx
        id1 = good_matches[j].queryIdx
        results[j, 0] = _kpts1[id1].pt[0]
        results[j, 1] = _kpts1[id1].pt[1]
        results[j, 2] = _kpts2[id2].pt[0]
        results[j, 3] = _kpts2[id2].pt[1]
    return results[:, 0:2], results[:, 2:]









