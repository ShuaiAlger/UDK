#
# Created  on 2020/6/27
#
from pathlib import Path
import argparse

import yaml
import numpy as np
from tqdm import tqdm

from hpatch_related.hpatch_dataset import OrgHPatchDataset

import cv2 as cv
import torch
import time

import os
import argparse
import yaml
import cv2
from DeepFeatureMatcher import DeepFeatureMatcher

from QDeepFeatureMatcher import QDeepFeatureMatcher
from SegDeepFeatureMatcher import SegDeepFeatureMatcher
from TransformerDeepFeatureMatcher import TRDeepFeatureMatcher
# from ManyDeepFeatureMatcher import ManyDeepFeatureMatcher
from ManyDFM import ManyDeepFeatureMatcher


from PIL import Image
import numpy as np
import time

def average_inference_time(time_collect):
    average_time = sum(time_collect) / len(time_collect)
    info = ('Average inference time: {}ms / {}fps'.format(
        round(average_time*1000), round(1/average_time))
    )
    print(info)
    return info


    # predictions = {
    #     "keypoints": XY[idxs],
    #     "descriptors": D[idxs],
    #     "scores": scores[idxs],
    #     "shape": shape
    # }

    # return predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,default='./configs/SDK_eva.yaml')
    parser.add_argument('--single', type=bool, default=True)
    # parser.add_argument('--output_root', type=str,default='hpatches_sequences/hpatches-sequences-release')
    parser.add_argument("--top-k", type=int, default=10000)
    parser.add_argument("--scale-f", type=float, default=2 ** 0.25)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=9999)
    parser.add_argument("--min-scale", type=float, default=0.3)
    parser.add_argument("--max-scale", type=float, default=1)
    parser.add_argument('--tag', type=str, default='re_mtldesc_v1')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    keys = '*' if config['keys'] == '*' else config['keys'].split(',')

    args.tag = str(config['model']['ckpt_name'])

    output_root = Path(config['hpatches']['dataset_dir'])    
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = OrgHPatchDataset(**config['hpatches'])


    model_name = "resnet18"


    manym = QDeepFeatureMatcher(enable_two_stage = False, model = model_name, 
                        ratio_th = [1.0], bidirectional = True)


    for i, data in tqdm(enumerate(dataset)):
        image_name = data['image_name']
        folder_name = data['folder_name']

        predictions = manym.extract_multiscale(data['image'])

        if config['output_type']=='benchmark':
            output_dir = Path(output_root,args.tag,folder_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            outpath = Path(output_dir, image_name)
            np.savez(str(outpath), **predictions)
        else:
            output_dir = Path(output_root, folder_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            outpath = Path(output_dir, image_name + '.ppm.' + args.tag)
            np.savez(open(outpath, 'wb'), **predictions)

     #   info = average_inference_time(time_collect)
      #  print(info)


