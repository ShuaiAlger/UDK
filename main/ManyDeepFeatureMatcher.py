#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:46:43 2021

@author: kutalmisince
"""
import numpy as np
import cv2 as cv
import torch
from torchvision import models, transforms
from collections import namedtuple
import matplotlib.pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor



import cv2



USE_CASCADE_RANSAC = 1

import copy

def draw_green(_img1, _img2, _m_pts1_src, _m_pts2_src):
    canvas = 255*np.ones((max(_img1.shape[0],_img2.shape[0]), _img1.shape[1]+_img2.shape[1], 3), dtype=np.uint8)

    _m_pts1 = copy.deepcopy(_m_pts1_src.copy())
    _m_pts2 = copy.deepcopy(_m_pts2_src.copy())

    canvas[0:_img1.shape[0], 0:_img1.shape[1], :] = _img1
    canvas[0:_img2.shape[0], _img1.shape[1]:_img1.shape[1]+_img2.shape[1], :] = _img2
    
    N_ = len(_m_pts1)

    for i in range(N_):
        color = (0, 255, 0)
        cv2.line(canvas, (int(_m_pts1[i, 0]), int(_m_pts1[i, 1])), (int(_m_pts2[i, 0]+_img1.shape[1]), int(_m_pts2[i, 1])), color, 1)
    return canvas


def desc2hist(desc):
    hist_A = np.asarray(desc * 255, dtype=np.uint8)
    hist_A_img = np.zeros((hist_A.shape[0], 256), dtype=np.uint8)
    for i in range(hist_A.shape[0]):
        hist_A_img[i, 255-hist_A[i]:255] = 255
    hist_A_img = hist_A_img.T
    return hist_A_img


def adaptive_image_pyramid(img, min_scale=0.0, max_scale=1, min_size=256, max_size=1536, scale_f=2**0.25, verbose=False):
    
    H, W, C = img.shape

    ## upsample the input to bigger size.
    s = 1.0
    if max(H, W) < max_size:
        s = max_size / max(H, W)
        max_scale = s
        nh, nw = round(H*s), round(W*s)
        # if verbose:  print(f"extracting at highest scale x{s:.02f} = {nw:4d}x{nh:3d}")
        # img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

        img = cv2.resize(img, (nw, nh))
        
    ## downsample the scale pyramid
    output = []
    scales = []
    while s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[0:2]

            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            output.append(img)
            scales.append(s)
        # print(f"passing the loop x{s:.02f} = {nw:4d}x{nh:3d}")        

        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = cv2.resize(img, (nw, nh))
    
    return output, scales

def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds





import math
import torch.nn.functional as F

def rot_fmap(tensor, angle):
    rad = angle/180*math.pi
    shift_x = 0.0
    shift_y = 0.0

    # 创建一个坐标变换矩阵
    transform_matrix = torch.tensor([
            [math.cos(rad), math.sin(-rad), shift_x],
            [math.sin(rad), math.cos(rad), shift_y]])

    # 将坐标变换矩阵的shape从[2,3]转化为[1,2,3]，并重复在第0维B次，最终shape为[B,2,3]
    transform_matrix = transform_matrix.unsqueeze(0).repeat(1, 1, 1)

    # print(tensor.shape, transform_matrix.shape)

    if (angle == 0) or (angle == 180) or (angle == -0) or (angle == -180):
        grid = F.affine_grid(transform_matrix, # 旋转变换矩阵
                                (1, tensor.shape[1], tensor.shape[2], tensor.shape[3]))	# 变换后的tensor的shape(与输入tensor相同)
    if (angle == 90) or (angle == 270) or (angle == -90) or (angle == -270):
        grid = F.affine_grid(transform_matrix, # 旋转变换矩阵
                                (1, tensor.shape[1], tensor.shape[3], tensor.shape[2]))	# 变换后的tensor的shape(与输入tensor相同)


    output = F.grid_sample(tensor, # 输入tensor，shape为[B,C,W,H]
                            grid, # 上一步输出的gird,shape为[B,C,W,H]
                            mode='bilinear') # 一些图像填充方法，这里我用的是最近邻


    # out_img = output[0, 0].cpu().numpy()

    # out_img = cv2.normalize(out_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # cv2.imshow("out_img", out_img)
    # cv2.waitKey(0)


    return output



def rot_seqs(image):
    image_rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    image_rot180 = cv2.rotate(image, cv2.ROTATE_180)

    image_rot270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image, image_rot90, image_rot180, image_rot270




from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models.efficientnet import efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.efficientnet import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s

from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.regnet import regnet_x_8gf, regnet_x_16gf, regnet_x_1_6gf, regnet_x_32gf, regnet_x_3_2gf, regnet_x_400mf, regnet_x_800mf

from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet50_2, wide_resnet101_2
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

from torchvision.models.resnet import resnext50_32x4d, resnext101_64x4d, resnext101_32x8d
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3

from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.convnext import convnext_base, convnext_large, convnext_small, convnext_tiny


from torchvision.models.swin_transformer import swin_b, swin_t, swin_s
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large


class ManyDeepFeatureMatcher(torch.nn.Module):
    def __init__(self, model: str = 'None', device = None, bidirectional=True, enable_two_stage = True,
                 ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0], layers = []):
        super(ManyDeepFeatureMatcher, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device == None else device        


        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)

        # model = "densenet121"       # 30.22 47.37 63.76 77.36
        # model = "densenet161"       # 29.79 46.84 63.52 78.00
        # model = "densenet169"       # 29.77 47.13 63.56 78.16
        
        # model = "wide_resnet50_2"   # 31.79 48.64 64.62 70.52
        # model = "wide_resnet101_2"  # 30.86 48.00 64.04 73.92

        # model = "efficientnet_b0"   # 24.90 39.85 55.87 64.76
        # model = "efficientnet_b3"   # 26.63 43.34 60.23 69.28
        # model = "efficientnet_b7"   # 27.33 44.38 60.76 73.97
        # model = "efficientnet_v2_s" # 27.72 44.54 60.98 68.16
        # model = "efficientnet_v2_l" # 26.37 42.10 58.74 70.47

        # model = "regnet_x_8gf"      # 21.30 35.95 51.77 54.92
        # model = "regnet_x_400mf"    # 20.57 35.01 51.73 59.96
        # model = "regnet_x_800mf"    # 24.69 39.48 55.53 65.51

        # model = "mobilenet_v2" # 19.53 33.17 49.13 48.21

        # model = "shufflenet_v2_x1_0" #  22.28 36.82 53.14 59.24
        
        # model = "deeplabv3_resnet50"            # 20.20 33.61 49.23 56.29
        # model = "deeplabv3_resnet101"           # 21.64 35.82 52.41 55.29
        # model = "deeplabv3_mobilenet_v3_large"  # 17.43 29.46 43.93 50.17

        # model = "regnet_x_16gf" #  5-layers     # 13.29 24.83 40.04 51.96
        # model = "regnet_x_16gf" #  4-layers     # 22.56 37.57 53.72 57.25
        # model = "regnet_x_1_6gf"                # 18.65 31.58 47.27 49.60

        # model = "regnet_x_32gf" # 22.47 37.27 53.47 57.78

                    # model = "resnet18" #  4-layers  # 24.30 39.00 55.14 58.59
                    # model = "resnet18" #  5-layers  # 25.89 41.65 58.03 65.98

        # model = "resnext50_32x4d"  #  30.58 47.10 62.86 71.23

        # model = "alexnet"  #  newest interpolation  # 22.03 36.17 51.87 70.99

        # model = "mnasnet1_0" # 22.55 37.13 52.42 65.31

        # model = "squeezenet1_0" # 18.90 33.38 50.16 70.54

        # model = "inception_v3" # 21.01 35.13 50.50 65.91



        model = "densenet121" # 



        self.model_name = model

        print("[self.model_name]: ", self.model_name)

        if model == "swin_b":
            self.return_nodes = { } #  18.85 32.31 48.01 66.84   (1.1.norm2, 3.1.norm2, 5.1.norm2, 7.0.norm2)
            self.return_nodes.update({'features.1.1.norm2':'features.1.1.norm2'})
            self.return_nodes.update({'features.3.1.norm2':'features.3.1.norm2'})
            self.return_nodes.update({'features.5.1.norm2':'features.5.1.norm2'})
            self.return_nodes.update({'features.7.1.norm2':'features.7.1.norm2'})
            self.smodel = swin_b(pretrained=True).to(self.device)
            print(self.smodel)

            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16

        if model == "swin_t":
            self.return_nodes = { } #  15.35 27.36 42.67 70.20   (1.1.norm2, 3.1.norm2, 5.1.norm2, 7.0.norm2)
            self.return_nodes.update({'features.1.1.norm2':'features.1.1.norm2'})
            self.return_nodes.update({'features.3.1.norm2':'features.3.1.norm2'})
            self.return_nodes.update({'features.5.1.norm2':'features.5.1.norm2'})
            self.return_nodes.update({'features.7.0.norm2':'features.7.0.norm2'})
            self.smodel = swin_t(pretrained=True).to(self.device)
            print(self.smodel)

            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16

        if model == "swin_s":
            self.return_nodes = { } #  12.88 24.54 38.59 64.32  (1.1.norm2, 3.1.norm2, 5.1.norm2, 7.0.norm2)
            self.return_nodes.update({'features.1.1.norm2':'features.1.1.norm2'})
            self.return_nodes.update({'features.3.1.norm2':'features.3.1.norm2'})
            self.return_nodes.update({'features.5.1.norm2':'features.5.1.norm2'})
            self.return_nodes.update({'features.7.0.norm2':'features.7.0.norm2'})
            self.smodel = swin_s(pretrained=True).to(self.device)
            print(self.smodel)

            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16


        if model == 'deeplabv3_resnet50':
            self.return_nodes = {
                'backbone.relu': 'backbone.relu',
                'backbone.layer1': 'backbone.layer1',
                'backbone.layer2': 'backbone.layer2',
                'backbone.layer3': 'backbone.layer3',
                'backbone.layer4.0': 'backbone.layer4.0',
            }
            self.smodel = deeplabv3_resnet50(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        if model == 'deeplabv3_resnet101':
            self.return_nodes = {
                'backbone.relu': 'backbone.relu',
                'backbone.layer1': 'backbone.layer1',
                'backbone.layer2': 'backbone.layer2',
                'backbone.layer3': 'backbone.layer3',
                'backbone.layer4.0': 'backbone.layer4.0',
            }
            self.smodel = deeplabv3_resnet101(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')


        if model == 'deeplabv3_mobilenet_v3_large':
            self.return_nodes = {
                'backbone.0': 'backbone.0',
                'backbone.2': 'backbone.2',
                'backbone.4': 'backbone.4',
                'backbone.7': 'backbone.7',
                'backbone.13': 'backbone.13',
            }
            self.smodel = deeplabv3_mobilenet_v3_large(pretrained=True).to(self.device)
            print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')


        if model == 'alexnet':
            self.return_nodes = { }
            self.return_nodes.update({'features.1':'features.1'})
            self.return_nodes.update({'features.4':'features.4'})
            self.return_nodes.update({'features.9':'features.9'})
            self.return_nodes.update({'features.12':'features.12'})

            self.smodel = alexnet(pretrained=True).to(self.device)

            # print(self.smodel)

            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet18':
            self.stage1 = [
                'relu'
            ]
            self.stage2 = [
                'maxpool',
                'layer1.0.relu',
                'layer1.0',
                'layer1.1.relu',
                'layer1.1'
            ]
            self.stage3 = [
                'layer2.0.relu',
                'layer2.0',
                'layer2.1.relu',
                'layer2.1'
            ]
            self.stage4 = [
                'layer3.0.relu',
                'layer3.0',
                'layer3.1.relu',
                'layer3.1'
            ]
            self.stage5 = [
                'layer4.0.relu',
                'layer4.0',
                'layer4.1.relu',
                'layer4.1'
            ]

            self.return_nodes_list = []

            for i1 in range(len(self.stage1)):
                for i2 in range(len(self.stage2)):
                    for i3 in range(len(self.stage3)):
                        for i4 in range(len(self.stage4)):
                            for i5 in range(len(self.stage5)):
                                self.return_nodes = {
                                    self.stage1[i1]: self.stage1[i1],
                                    self.stage2[i2]: self.stage2[i2],
                                    self.stage3[i3]: self.stage3[i3],
                                    self.stage4[i4]: self.stage4[i4],
                                    self.stage5[i5]: self.stage5[i5],
                                }
                                self.return_nodes_list.append(self.return_nodes)


            print(len(self.return_nodes_list))


            self.return_nodes = {
                self.stage1[0]: self.stage1[0],
                self.stage2[0]: self.stage2[0],
                self.stage3[0]: self.stage3[0],
                self.stage4[0]: self.stage4[0],
                self.stage5[0]: self.stage5[0],
            }

            self.smodel = resnet18(pretrained=True).to(self.device)
            
            print(self.smodel)
            
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet34':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnet34(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet50':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnet50(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet101':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnet101(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnet152':
            self.return_nodes = {
                'relu': 'relu',
                'layer1.0': 'layer1.0',
                'layer2.0': 'layer2.0',
                'layer3.0': 'layer3.0',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnet152(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'wide_resnet50_2':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4.0': 'layer4.0',
            }
            self.smodel = wide_resnet50_2(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'wide_resnet101_2':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4.0': 'layer4.0',
            }
            self.smodel = wide_resnet101_2(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')


        elif model == 'resnext50_32x4d':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2.0': 'layer2.0',
                'layer3.0': 'layer3.0',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnext50_32x4d(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnext101_32x8d':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2.0': 'layer2.0',
                'layer3.0': 'layer3.0',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnext101_32x8d(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'resnext101_64x4d':
            self.return_nodes = {
                'relu': 'relu',
                'layer1': 'layer1',
                'layer2.0': 'layer2.0',
                'layer3.0': 'layer3.0',
                'layer4.0': 'layer4.0',
            }
            self.smodel = resnext101_64x4d(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')




        elif model == 'squeezenet1_0':
            self.return_nodes = {
                'features.1': 'features.1',
                'features.5': 'features.5',
                'features.10': 'features.10',
                'features.11': 'features.11',
            }
            self.smodel = squeezenet1_0(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'squeezenet1_1':
            self.return_nodes = {
                'features.1': 'features.1',
                'features.3': 'features.3',
                'features.6': 'features.6',
                'features.11': 'features.11',
            }
            self.smodel = squeezenet1_1(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')



        elif model == 'googlenet':
            self.return_nodes = {
                'conv1': 'conv1',
                'maxpool1': 'maxpool1',
                'maxpool2': 'maxpool2',
                'maxpool3': 'maxpool3',
                'maxpool4': 'maxpool4',
            }
            self.qmodel = googlenet(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'inception_v3':
            self.return_nodes = {
                'Conv2d_1a_3x3': 'Conv2d_1a_3x3',
                'maxpool1': 'maxpool1',
                'maxpool2': 'maxpool2',
                'Mixed_6a.branch3x3': 'Mixed_6a.branch3x3',
                'Mixed_7a.branch3x3_2': 'Mixed_7a.branch3x3_2',
            }
            self.qmodel = inception_v3(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mobilenet_v2':
            self.return_nodes = {
                'features.1': 'features.1',
                'features.3.conv.0': 'features.3.conv.0',
                'features.6.conv.1': 'features.6.conv.1',
                'features.13.conv.0': 'features.13.conv.0',
                'features.17.conv.0': 'features.17.conv.0',
            }
            self.smodel = mobilenet_v2(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mobilenet_v3_small':
            self.return_nodes = {
                'features.0': 'features.0',
                'features.1': 'features.1',
                'features.3': 'features.3',
                'features.8': 'features.8',
                'features.10': 'features.10',
            }
            self.smodel = mobilenet_v3_small(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mobilenet_v3_large':
            self.return_nodes = {
                'features.0': 'features.0',
                'features.2': 'features.2',
                'features.4': 'features.4',
                'features.7': 'features.7',
                'features.13': 'features.13',
            }
            self.qmodel = mobilenet_v3_large(pretrained=True, quantize=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')


        elif model == 'vgg11': # 24.66 40.30 56.52 59.40 # ShiTomasi
            self.return_nodes = {
                'features.1': 'features.1',
                'features.4': 'features.4',
                'features.7': 'features.7',
                'features.12': 'features.12',
                'features.17': 'features.17',
            }
            self.smodel = vgg11(pretrained=True).to(self.device)
            print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'vgg13': # 25.91 42.49 59.14 60.37 # ShiTomasi
            self.return_nodes = {
                'features.1': 'features.1',
                'features.6': 'features.6',
                'features.11': 'features.11',
                'features.16': 'features.16',
                'features.21': 'features.21',
            }
            self.smodel = vgg13(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'vgg16': # 25.33 41.70 58.02 60.40 # ShiTomasi
            self.return_nodes = {
                'features.1': 'features.1',
                'features.6': 'features.6',
                'features.11': 'features.11',
                'features.18': 'features.18',
                'features.25': 'features.25',
            }
            self.smodel = vgg16(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'vgg19': # 21.24 36.69 53.10 52.19 # ShiTomasi
            self.return_nodes = {
                'features.1': 'features.1',
                'features.6': 'features.6',
                'features.11': 'features.11',
                'features.18': 'features.18',
                'features.25': 'features.25',
            }
            
            self.smodel = vgg19(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            # self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')




        elif model == 'shufflenet_v2_x0_5':
            self.return_nodes = {
                'conv1': 'conv1',
                'maxpool': 'maxpool',
                'stage2.0': 'stage2.0',
                'stage3.0': 'stage3.0',
                'stage4.0': 'stage4.0',
            }
            self.qmodel = shufflenet_v2_x0_5(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'shufflenet_v2_x1_0':
            self.return_nodes = {
                'conv1': 'conv1',
                'maxpool': 'maxpool',
                'stage2.0': 'stage2.0',
                'stage3.0': 'stage3.0',
                'stage4.0': 'stage4.0',
            }
            self.qmodel = shufflenet_v2_x1_0(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'shufflenet_v2_x1_5':
            self.return_nodes = {
                'conv1': 'conv1',
                'maxpool': 'maxpool',
                'stage2.0': 'stage2.0',
                'stage3.0': 'stage3.0',
                'stage4.0': 'stage4.0',
            }
            self.qmodel = shufflenet_v2_x1_5(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'shufflenet_v2_x2_0':
            self.return_nodes = {
                'conv1': 'conv1',
                'maxpool': 'maxpool',
                'stage2.0': 'stage2.0',
                'stage3.0': 'stage3.0',
                'stage4.0': 'stage4.0',
            }
            self.qmodel = shufflenet_v2_x2_0(pretrained=True).to(self.device)
            print(self.qmodel)
            self.model = create_feature_extractor(self.qmodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')




        elif model == 'densenet121':
            self.return_nodes = { }
            self.return_nodes.update({'features.relu0':'features.relu0'})
            self.return_nodes.update({'features.pool0':'features.pool0'})
            self.return_nodes.update({'features.denseblock2':'features.denseblock2'})
            self.return_nodes.update({'features.denseblock3':'features.denseblock3'})
            self.return_nodes.update({'features.denseblock4':'features.denseblock4'})

            self.smodel = densenet121(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'densenet161':
            self.return_nodes = { }
            self.return_nodes.update({'features.relu0':'features.relu0'})
            self.return_nodes.update({'features.pool0':'features.pool0'})
            self.return_nodes.update({'features.denseblock2':'features.denseblock2'})
            self.return_nodes.update({'features.denseblock3':'features.denseblock3'})
            self.return_nodes.update({'features.denseblock4':'features.denseblock4'})

            self.smodel = densenet161(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'densenet169':
            self.return_nodes = { }
            self.return_nodes.update({'features.relu0':'features.relu0'})
            self.return_nodes.update({'features.pool0':'features.pool0'})
            self.return_nodes.update({'features.denseblock2':'features.denseblock2'})
            self.return_nodes.update({'features.denseblock3':'features.denseblock3'})
            self.return_nodes.update({'features.denseblock4':'features.denseblock4'})

            self.smodel = densenet169(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'densenet201':
            self.return_nodes = { }
            self.return_nodes.update({'features.relu0':'features.relu0'})
            self.return_nodes.update({'features.pool0':'features.pool0'})
            self.return_nodes.update({'features.denseblock2':'features.denseblock2'})
            self.return_nodes.update({'features.denseblock3':'features.denseblock3'})
            self.return_nodes.update({'features.denseblock4':'features.denseblock4'})

            self.smodel = densenet201(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')






        elif model == 'convnext_base':
            self.return_nodes = { }
            self.return_nodes.update({'features.1.0':'features.1.0'})
            self.return_nodes.update({'features.2':'features.2'})
            self.return_nodes.update({'features.4':'features.4'})
            self.return_nodes.update({'features.6':'features.6'})

            self.smodel = convnext_base(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'convnext_large':
            self.return_nodes = { }
            self.return_nodes.update({'features.1.2':'features.1.2'})
            self.return_nodes.update({'features.2':'features.2'})
            self.return_nodes.update({'features.4':'features.4'})
            self.return_nodes.update({'features.6':'features.6'})

            self.smodel = convnext_large(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'convnext_small':
            self.return_nodes = { }
            self.return_nodes.update({'features.1.0':'features.1.0'})
            self.return_nodes.update({'features.2':'features.2'})
            self.return_nodes.update({'features.4':'features.4'})
            self.return_nodes.update({'features.6':'features.6'})

            self.smodel = convnext_small(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'convnext_tiny':
            self.return_nodes = { }
            self.return_nodes.update({'features.1.0':'features.1.0'})
            self.return_nodes.update({'features.2':'features.2'})
            self.return_nodes.update({'features.4':'features.4'})
            self.return_nodes.update({'features.6':'features.6'})

            self.smodel = convnext_tiny(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')


        elif model == 'regnet_x_8gf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_8gf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_16gf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_16gf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_1_6gf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_1_6gf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_32gf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_32gf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_3_2gf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_3_2gf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_400mf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_400mf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'regnet_x_800mf':
            self.return_nodes = { }
            self.return_nodes.update({'stem':'stem'})
            self.return_nodes.update({'trunk_output.block1.block1-0.f':'trunk_output.block1.block1-0.f'})
            self.return_nodes.update({'trunk_output.block2.block2-0.f':'trunk_output.block2.block2-0.f'})
            self.return_nodes.update({'trunk_output.block3.block3-0.f':'trunk_output.block3.block3-0.f'})
            self.return_nodes.update({'trunk_output.block4.block4-0.f':'trunk_output.block4.block4-0.f'})

            self.smodel = regnet_x_800mf(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')



        elif model == 'efficientnet_b0':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b0(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b1':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b1(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b2':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b2(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b3':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b3(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b4':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b4(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')
      
        elif model == 'efficientnet_b5':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b5(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b6':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b6(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_b7':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_b7(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_v2_s':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_v2_s(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_v2_m':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_v2_m(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'efficientnet_v2_l':
            self.return_nodes = { }
            self.return_nodes.update({'features.0':'features.0'})
            self.return_nodes.update({'features.2.0':'features.2.0'})
            self.return_nodes.update({'features.3.0':'features.3.0'})
            self.return_nodes.update({'features.5.0':'features.5.0'})
            self.return_nodes.update({'features.6.0':'features.6.0'})

            self.smodel = efficientnet_v2_l(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')







        elif model == 'mnasnet0_5':
            self.return_nodes = { }
            self.return_nodes.update({'layers.2':'layers.2'})
            self.return_nodes.update({'layers.8.0':'layers.8.0'})
            self.return_nodes.update({'layers.9.0':'layers.9.0'})
            self.return_nodes.update({'layers.10.0':'layers.10.0'})
            self.return_nodes.update({'layers.12.0':'layers.12.0'})

            self.smodel = mnasnet0_5(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mnasnet0_75':
            self.return_nodes = { }
            self.return_nodes.update({'layers.2':'layers.2'})
            self.return_nodes.update({'layers.8.0':'layers.8.0'})
            self.return_nodes.update({'layers.9.0':'layers.9.0'})
            self.return_nodes.update({'layers.10.0':'layers.10.0'})
            self.return_nodes.update({'layers.12.0':'layers.12.0'})

            self.smodel = mnasnet0_75(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mnasnet1_0':
            self.return_nodes = { }
            self.return_nodes.update({'layers.2':'layers.2'})
            self.return_nodes.update({'layers.8.0':'layers.8.0'})
            self.return_nodes.update({'layers.9.0':'layers.9.0'})
            self.return_nodes.update({'layers.10.0':'layers.10.0'})
            self.return_nodes.update({'layers.12.0':'layers.12.0'})

            self.smodel = mnasnet1_0(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')

        elif model == 'mnasnet1_3':
            self.return_nodes = { }
            self.return_nodes.update({'layers.2':'layers.2'})
            self.return_nodes.update({'layers.8.0':'layers.8.0'})
            self.return_nodes.update({'layers.9.0':'layers.9.0'})
            self.return_nodes.update({'layers.10.0':'layers.10.0'})
            self.return_nodes.update({'layers.12.0':'layers.12.0'})

            self.smodel = mnasnet1_3(pretrained=True).to(self.device)
            # print(self.smodel)
            self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
            self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.padding_n = 16
            print('model is loaded.')
        else:
            print('Error: model ' + model + ' is not supported!')
            return
        
        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)





    # def match(self, img_A, img_B, display_results=0, *args):

    #     maphA, mapwA = img_A.shape[0], img_A.shape[1]
    #     maphB, mapwB = img_B.shape[0], img_B.shape[1]


    #     block_size = 3
    #     sobel_size = 3
    #     k = 0.06

    #     grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    #     grayA = np.float32(grayA)

    #     grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
    #     grayB = np.float32(grayB)




    #     corners_imgA = cv2.cornerHarris(grayA, block_size, sobel_size, k)
    #     corners_imgB = cv2.cornerHarris(grayB, block_size, sobel_size, k)


    #     cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.05, 3) 
    #     cornersB = cv2.goodFeaturesToTrack(grayB, 10000, 0.05, 3) 



    #     # transform into pytroch tensor and pad image to a multiple of 16
    #     inp_A, padding_A = self.transform(img_A) 
    #     inp_B, padding_B = self.transform(img_B)

    #     outputs_A = self.model(inp_A)
    #     outputs_B = self.model(inp_B)

    #     activations_A = []
    #     activations_B = []


    #     for x in self.return_nodes:
    #         activations_A.append(outputs_A[str(x)])
    #         activations_B.append(outputs_B[str(x)])

    #     if (self.model_name[0] == "v") and (self.model_name[1] == "g") and (self.model_name[2] == "g"):
    #         pass
    #     else:
    #         for i in range(len(activations_A)):
    #             activations_A[i] = self.upsample(activations_A[i])
    #             activations_B[i] = self.upsample(activations_B[i])


    #         # print(activations_A[i].shape)



    #     maph, mapw = activations_A[0].shape[2], activations_A[0].shape[3]
    #     acA = activations_A[-4][0, :].detach().cpu().numpy()
    #     acA = np.sum(acA, axis=0)
    #     acA = cv2.normalize(acA, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    #     acA = cv2.resize(acA, (mapw, maph))
    #     # detectionA3 = cv2.applyColorMap(acA, cv2.COLORMAP_JET)
        
    #     maph, mapw = activations_B[0].shape[2], activations_B[0].shape[3]

    #     acB = activations_B[-4][0, :].detach().cpu().numpy()
    #     acB = np.sum(acB, axis=0)


    #     acB = cv2.normalize(acB, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    #     acB = cv2.resize(acB, (mapw, maph))
    #     # detectionB3 = cv2.applyColorMap(acB, cv2.COLORMAP_JET)

    #     detectionA3 = np.zeros_like(acA, dtype=np.float64)
    #     detectionB3 = np.zeros_like(acB, dtype=np.float64)


    #     corners_imgA = cv2.resize(corners_imgA, (detectionA3.shape[1], detectionA3.shape[0]))
    #     corners_imgB = cv2.resize(corners_imgB, (detectionB3.shape[1], detectionB3.shape[0]))

    #     detectionA3[corners_imgA > np.percentile(corners_imgA, 98)] = 1
    #     detectionB3[corners_imgB > np.percentile(corners_imgB, 98)] = 1


    #     # detectionA3[acA < np.percentile(acA, 3)] = 1
    #     # detectionB3[acB < np.percentile(acB, 3)] = 1

    #     # detectionA3[acA > np.percentile(acA, 99.5)] = 1
    #     # detectionB3[acB > np.percentile(acB, 99.5)] = 1


    #     if 0:
    #         detectionA = np.argwhere(detectionA3 > 0)
    #         detectionB = np.argwhere(detectionB3 > 0)

    #     else:
    #         detectionA = np.roll(cornersA[:, 0, :], 1, axis=-1)
    #         detectionB = np.roll(cornersB[:, 0, :], 1, axis=-1)

    #     descriptorsA = activations_A[0][0][:, detectionA[:, 0], detectionA[:, 1]]
    #     descriptorsB = activations_B[0][0][:, detectionB[:, 0], detectionB[:, 1]]

    #     if 1:
    #         descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)

    #         descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:, np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)

    #         descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)

    #         # descriptorsA = torch.concat([descriptorsA, activations_A[4][0][0:, np.int64(detectionA[:, 0]/16), np.int64(detectionA[:, 1]/16)]], dim=0)
    #         # descriptorsB = torch.concat([descriptorsB, activations_B[4][0][0:, np.int64(detectionB[:, 0]/16), np.int64(detectionB[:, 1]/16)]], dim=0)

    #     else:
    #         descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)

    #         descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0: , np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0: , np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)

    #         descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0: , np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
    #         descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0: , np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)

    #         # descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:128, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
    #         # descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:128, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)



    #     matches, scores = dense_feature_matching(descriptorsA, descriptorsB, 1, True)


    #     pointsA = detectionA[matches[:, 0]]
    #     pointsB = detectionB[matches[:, 1]]

    #     pointsA = np.roll(pointsA, 1, axis=-1)
    #     pointsB = np.roll(pointsB, 1, axis=-1)





    #     canvas = draw_green(img_A, img_B, pointsA, pointsB)



    #     detectionA3 = cv2.normalize(detectionA3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #     detectionB3 = cv2.normalize(detectionB3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #     detectionA3 = cv2.cvtColor(detectionA3, cv2.COLOR_GRAY2BGR)
    #     detectionB3 = cv2.cvtColor(detectionB3, cv2.COLOR_GRAY2BGR)

    #     img_A = cv2.resize(img_A, (detectionA3.shape[1], detectionA3.shape[0]))
    #     img_B = cv2.resize(img_B, (detectionB3.shape[1], detectionB3.shape[0]))



    #     # if USE_CASCADE_RANSAC:
    #     #     if len(pointsA) > 8:
    #     #         model, inliers = pydegensac.findFundamentalMatrix(pointsA, pointsB, 3.0, 0.9999, 100000)
    #     #         pointsA, pointsB = pointsA[inliers], pointsB[inliers]




    #     canvas2 = draw_green(img_A*detectionA3, img_B*detectionB3, pointsA, pointsB)


    #     cv2.namedWindow("canvas", 0)
    #     cv2.imshow("canvas", canvas)

    #     cv2.namedWindow("canvas2", 0)
    #     cv2.imshow("canvas2", canvas2)

    #     cv2.waitKey(1)

    #     return pointsA, pointsB





    # def extract_single(self, img_A):
    #     grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    #     grayA = np.float32(grayA)

    #     cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.01, 5) 

    #     img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)

    #     inp_A, padding_A = self.transform(img_A) 

    #     outputs_A = self.model(inp_A)

    #     activations_A = []


    #     for x in self.return_nodes:
    #         activations_A.append(outputs_A[str(x)])

    #     if (self.model_name[0] == "v") and (self.model_name[1] == "g") and (self.model_name[2] == "g"):
    #         pass
    #     else:
    #         for i in range(len(activations_A)):
    #             activations_A[i] = self.upsample(activations_A[i])

    #     detectionA = np.roll(cornersA[:, 0, :], 1, axis=-1)


    #     descriptorsA = activations_A[0][0][:, detectionA[:, 0], detectionA[:, 1]]


    #     descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)

    #     descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)

    #     descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)

    #     descriptorsA = torch.concat([descriptorsA, activations_A[4][0][0:, np.int64(detectionA[:, 0]/16), np.int64(detectionA[:, 1]/16)]], dim=0)


    #     return detectionA, descriptorsA


    # def extract_multiscale(self, img_A):

    #     maphA, mapwA = img_A.shape[0], img_A.shape[1]

    #     imagesA, scalesA = adaptive_image_pyramid(img_A, max_size=1200)


    #     pointsAs = []
    #     descriptorsAs = []


    #     for img_A, scale_A in zip(imagesA, scalesA):
    #         pointsA, descriptorsA = self.extract_single(img_A)
    #         pointsAs.append(pointsA / scale_A)
    #         descriptorsAs.append(descriptorsA)

    #     # pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7]], axis=0)
    #     # descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7]], dim=1)

    #     pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7],
    #                                  pointsAs[1], pointsAs[3], pointsAs[4], pointsAs[6]], axis=0)
    #     descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7],
    #                                   descriptorsAs[1], descriptorsAs[3], descriptorsAs[4], descriptorsAs[6]], dim=1)


    #     pointsA = np.roll(pointsAall, 1, axis=-1)


    #     # return pointsA, pointsB

    #     predictions = {
    #         "keypoints": pointsA,
    #         "descriptors": descriptorsAsall,
    #         "scores": np.ones_like(pointsA[:, 0]),
    #         "shape": img_A.shape
    #     }

    #     return predictions






    def extract_singlescale(self, img_A, img_B, display_results=0, *args):
        grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        
        cornersA = None
        cornersB = None

        self.USE_FAST = 0
        self.USE_SHITomasi = 1

        if self.USE_FAST:
            fast = cv2.FastFeatureDetector_create(threshold=12, nonmaxSuppression=True)
            # find and draw the keypoints
            fastA = fast.detect(grayA, None)
            fastB = fast.detect(grayB, None)

            cornersA = np.zeros((len(fastA), 1, 2))
            cornersB = np.zeros((len(fastB), 1, 2))
            for i in range(len(fastA)):
                cornersA[i, 0, 0] = fastA[i].pt[0]
                cornersA[i, 0, 1] = fastA[i].pt[1]

            for i in range(len(fastB)):
                cornersB[i, 0, 0] = fastB[i].pt[0]
                cornersB[i, 0, 1] = fastB[i].pt[1]

        else:
            grayA = np.float32(grayA)
            grayB = np.float32(grayB)
            cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.03, 5) 
            cornersB = cv2.goodFeaturesToTrack(grayB, 10000, 0.03, 5) 

            # cornersA = cv2.goodFeaturesToTrack(grayA, 5000, 0.01, 3) 
            # cornersB = cv2.goodFeaturesToTrack(grayB, 5000, 0.01, 3) 

        img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
        img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)

        hA, wA = img_A.shape[0], img_A.shape[1]
        hB, wB = img_B.shape[0], img_B.shape[1]

        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B)

        newhA, newwA = inp_A.shape[-2], inp_A.shape[-1]
        newhB, newwB = inp_B.shape[-2], inp_B.shape[-1]

        outputs_A = self.model(inp_A)
        outputs_B = self.model(inp_B)

        activations_A = []
        activations_B = []


        for x in self.return_nodes:
            if (self.model_name[0] == "s") and (self.model_name[1] == "w") and (self.model_name[2] == "i"):
                activations_A.append(outputs_A[str(x)].transpose(1, 3).transpose(2, 3)) # swin
                activations_B.append(outputs_B[str(x)].transpose(1, 3).transpose(2, 3)) # swin
            else:
                activations_A.append(outputs_A[str(x)])
                activations_B.append(outputs_B[str(x)])


            # print("outputs_A", str(x), ".shape : ", outputs_A[str(x)].shape)

        if (self.model_name[0] == "v") and (self.model_name[1] == "g") and (self.model_name[2] == "g"):
            pass
        else:
            for i in range(len(activations_A)):
                activations_A[i] = self.upsample(activations_A[i])
                activations_B[i] = self.upsample(activations_B[i])

        detectionA = np.roll(cornersA[:, 0, :], 1, axis=-1)
        detectionB = np.roll(cornersB[:, 0, :], 1, axis=-1)


        ah, aw = activations_A[0][0].shape[1], activations_A[0][0].shape[2]
        bh, bw = activations_B[0][0].shape[1], activations_B[0][0].shape[2]

        ratioha = hA/ah
        ratiowa = wA/aw
        ratiohb = hB/bh
        ratiowb = wB/bw

        descriptorsA = activations_A[0][0][:, np.int64(detectionA[:, 0]/ratioha), np.int64(detectionA[:, 1]/ratiowa)]
        descriptorsB = activations_B[0][0][:, np.int64(detectionB[:, 0]/ratiohb), np.int64(detectionB[:, 1]/ratiowb)]

        ah, aw = activations_A[1][0].shape[1], activations_A[1][0].shape[2]
        bh, bw = activations_B[1][0].shape[1], activations_B[1][0].shape[2]

        ratioha = hA/2/ah
        ratiowa = wA/2/aw
        ratiohb = hB/2/bh
        ratiowb = wB/2/bw

        descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/ratioha/2), np.int64(detectionA[:, 1]/ratiowa/2)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/ratiohb/2), np.int64(detectionB[:, 1]/ratiowb/2)]], dim=0)

        ah, aw = activations_A[2][0].shape[1], activations_A[2][0].shape[2]
        bh, bw = activations_B[2][0].shape[1], activations_B[2][0].shape[2]

        ratioha = hA/4/ah
        ratiowa = wA/4/aw
        ratiohb = hB/4/bh
        ratiowb = wB/4/bw

        descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/ratioha/4), np.int64(detectionA[:, 1]/ratiowa/4)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:, np.int64(detectionB[:, 0]/ratiohb/4), np.int64(detectionB[:, 1]/ratiowb/4)]], dim=0)


        self.USE_Layer4 = 1
        if self.USE_Layer4:
            ah, aw = activations_A[3][0].shape[1], activations_A[3][0].shape[2]
            bh, bw = activations_B[3][0].shape[1], activations_B[3][0].shape[2]

            ratioha = hA/8/ah
            ratiowa = wA/8/aw
            ratiohb = hB/8/bh
            ratiowb = wB/8/bw
            descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:, np.int64(detectionA[:, 0]/ratioha/8), np.int64(detectionA[:, 1]/ratiowa/8)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:, np.int64(detectionB[:, 0]/ratiohb/8), np.int64(detectionB[:, 1]/ratiowb/8)]], dim=0)


        self.USE_Layer5 = 0
        if self.USE_Layer5:
            ah, aw = activations_A[4][0].shape[1], activations_A[4][0].shape[2]
            bh, bw = activations_B[4][0].shape[1], activations_B[4][0].shape[2]

            ratioha = hA/16/ah
            ratiowa = wA/16/aw
            ratiohb = hB/16/bh
            ratiowb = wB/16/bw
            descriptorsA = torch.concat([descriptorsA, activations_A[4][0][0:, np.int64(detectionA[:, 0]/ratioha/16), np.int64(detectionA[:, 1]/ratiowa/16)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[4][0][0:, np.int64(detectionB[:, 0]/ratiohb/16), np.int64(detectionB[:, 1]/ratiowb/16)]], dim=0)


        return detectionA, detectionB, descriptorsA, descriptorsB





    def match_multiscale(self, img_A, img_B, display_results=0, *args):

        with torch.no_grad():
            maphA, mapwA = img_A.shape[0], img_A.shape[1]
            maphB, mapwB = img_B.shape[0], img_B.shape[1]

            imagesA, scalesA = adaptive_image_pyramid(img_A, max_size=1200)
            imagesB, scalesB = adaptive_image_pyramid(img_B, max_size=1200)

            pointsAs = []
            pointsBs = []
            descriptorsAs = []
            descriptorsBs = []

            for img_A, img_B, scale_A, scale_B in zip(imagesA, imagesB, scalesA, scalesB):
                pointsA, pointsB, descriptorsA, descriptorsB = self.extract_singlescale(img_A, img_B)
                pointsAs.append(pointsA / scale_A)
                pointsBs.append(pointsB / scale_B)
                descriptorsAs.append(descriptorsA)
                descriptorsBs.append(descriptorsB)

            self.USE_1_SCALE = 0
            self.USE_4_MULTI_SCALE = 0

            if self.USE_1_SCALE:
                pointsAall = pointsAs[7]
                pointsBall = pointsBs[7]
                descriptorsAsall = descriptorsAs[7]
                descriptorsBsall = descriptorsBs[7]

            elif self.USE_4_MULTI_SCALE:
                pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7]], axis=0)
                pointsBall = np.concatenate([pointsBs[0], pointsBs[2], pointsBs[5], pointsBs[7]], axis=0)
                descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7]], dim=1)
                descriptorsBsall = torch.cat([descriptorsBs[0], descriptorsBs[2], descriptorsBs[5], descriptorsBs[7]], dim=1)
            else:
                pointsAall = np.concatenate([pointsAs[0], pointsAs[2], pointsAs[5], pointsAs[7],
                                            pointsAs[1], pointsAs[3], pointsAs[4], pointsAs[6]], axis=0)
                pointsBall = np.concatenate([pointsBs[0], pointsBs[2], pointsBs[5], pointsBs[7],
                                            pointsBs[1], pointsBs[3], pointsBs[4], pointsBs[6]], axis=0)
                descriptorsAsall = torch.cat([descriptorsAs[0], descriptorsAs[2], descriptorsAs[5], descriptorsAs[7],
                                            descriptorsAs[1], descriptorsAs[3], descriptorsAs[4], descriptorsAs[6]], dim=1)
                descriptorsBsall = torch.cat([descriptorsBs[0], descriptorsBs[2], descriptorsBs[5], descriptorsBs[7],
                                            descriptorsBs[1], descriptorsBs[3], descriptorsBs[4], descriptorsBs[6]], dim=1)

            matches, scores = dense_feature_matching(descriptorsAsall, descriptorsBsall, 0.999, True)

            pointsA = pointsAall[matches[:, 0]]
            pointsB = pointsBall[matches[:, 1]]

            pointsA = np.roll(pointsA, 1, axis=-1)
            pointsB = np.roll(pointsB, 1, axis=-1)

            return pointsA, pointsB
















    def extract_descriptor(self, img_A, img_B):
        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B)

        outputs_A = self.model(inp_A)
        outputs_B = self.model(inp_B)

        activations_A = []
        activations_B = []


        for x in self.return_nodes:
            activations_A.append(outputs_A[str(x)])
            activations_B.append(outputs_B[str(x)])

        for i in range(len(activations_A)):
            activations_A[i] = self.upsample(activations_A[i])
            activations_B[i] = self.upsample(activations_B[i])
        return activations_A, activations_B


    def combine_desc(self, activations_A, activations_B, detectionA, detectionB):
        descriptorsA = activations_A[0][0][:, detectionA[:, 0], detectionA[:, 1]]
        descriptorsB = activations_B[0][0][:, detectionB[:, 0], detectionB[:, 1]]


        descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)

        descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:, np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)

        descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)

        descriptorsA = torch.concat([descriptorsA, activations_A[4][0][0:, np.int64(detectionA[:, 0]/16), np.int64(detectionA[:, 1]/16)]], dim=0)
        descriptorsB = torch.concat([descriptorsB, activations_B[4][0][0:, np.int64(detectionB[:, 0]/16), np.int64(detectionB[:, 1]/16)]], dim=0)
        return descriptorsA, descriptorsB


    def extract_rotation(self, img_A, img_B, display_results=0, *args):
        grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        grayA = np.float32(grayA)

        grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        grayB = np.float32(grayB)

        cornersA = cv2.goodFeaturesToTrack(grayA, 10000, 0.03, 5) 
        cornersB = cv2.goodFeaturesToTrack(grayB, 10000, 0.03, 5) 

        detectionA = np.roll(cornersA[:, 0, :], 1, axis=-1)
        detectionB = np.roll(cornersB[:, 0, :], 1, axis=-1)

        img_A, img_Arot90, img_Arot180, img_Arot270 = rot_seqs(img_A)
        img_B, img_Brot90, img_Brot180, img_Brot270 = rot_seqs(img_B)


        activations_A, activations_B = self.extract_descriptor(img_A, img_B)
        activations_Arot90, activations_Brot90 = self.extract_descriptor(img_Arot90, img_Brot90)
        activations_Arot180, activations_Brot180 = self.extract_descriptor(img_Arot180, img_Brot180)
        activations_Arot270, activations_Brot270 = self.extract_descriptor(img_Arot270, img_Brot270)


        # print(activations_A[0].shape, activations_Arot90[0].shape, activations_Arot180[0].shape, activations_Arot270[0].shape)


        for i in range(len(activations_Arot90)):
            activations_Arot90[i] = rot_fmap(activations_Arot90[i], 90)
            activations_Brot90[i] = rot_fmap(activations_Brot90[i], 90)
            activations_Arot180[i] = rot_fmap(activations_Arot180[i], 180)
            activations_Brot180[i] = rot_fmap(activations_Brot180[i], 180)
            activations_Arot270[i] = rot_fmap(activations_Arot270[i], 270)
            activations_Brot270[i] = rot_fmap(activations_Brot270[i], 270)

        
        # print(activations_A[0].shape)


        # img0 = activations_A[0][0].norm(dim=0).cpu().numpy()
        # img1 = activations_Arot90[0][0].norm(dim=0).cpu().numpy()
        # img2 = activations_Arot180[0][0].norm(dim=0).cpu().numpy()
        # img3 = activations_Arot270[0][0].norm(dim=0).cpu().numpy()

        # img0 = activations_B[0][0].norm(dim=0).cpu().numpy()
        # img1 = activations_Brot90[0][0].norm(dim=0).cpu().numpy()
        # img2 = activations_Brot180[0][0].norm(dim=0).cpu().numpy()
        # img3 = activations_Brot270[0][0].norm(dim=0).cpu().numpy()

        # img0 = cv2.normalize(img0, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # img1 = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # img2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # img3 = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # cv2.imshow("img0", img0)
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)
        # cv2.imshow("img3", img3)

        # cv2.waitKey(1)



        # print(activations_A[0].shape, activations_Arot90[0].shape, activations_Arot180[0].shape, activations_Arot270[0].shape)

        descriptorsA, descriptorsB = self.combine_desc(activations_A, activations_B, detectionA, detectionB)
        descriptorsArot90, descriptorsBrot90 = self.combine_desc(activations_Arot90, activations_Brot90, detectionA, detectionB)
        descriptorsArot180, descriptorsBrot180 = self.combine_desc(activations_Arot180, activations_Brot180, detectionA, detectionB)
        descriptorsArot270, descriptorsBrot270 = self.combine_desc(activations_Arot270, activations_Brot270, detectionA, detectionB)



        descriptorsA = torch.cat([descriptorsA, descriptorsArot90, descriptorsArot180, descriptorsArot270], dim=0)
        descriptorsB = torch.cat([descriptorsB, descriptorsBrot90, descriptorsBrot180, descriptorsBrot270], dim=0)

        return detectionA, detectionB, descriptorsA, descriptorsB

    def match_rotation(self, img_A, img_B, display_results=0, *args):

        pointsAall, pointsBall, descriptorsAsall, descriptorsBsall = self.extract_rotation(img_A, img_B)

        sdim = int(descriptorsAsall.shape[0]/4)

        

        for i in range(descriptorsAsall.shape[1]):
            desc_rot0 = descriptorsAsall[0*sdim:1*sdim, i]
            desc_rot1 = descriptorsAsall[1*sdim:2*sdim, i]
            desc_rot2 = descriptorsAsall[2*sdim:3*sdim, i]
            desc_rot3 = descriptorsAsall[3*sdim:4*sdim, i]

            norm_res = torch.tensor([desc_rot0.sum(), desc_rot1.sum(), desc_rot2.sum(), desc_rot3.sum()])

            max_id = torch.argmax(norm_res)
            if max_id == 1:
                descriptorsAsall[:, i] = torch.concat([desc_rot1, desc_rot2, desc_rot3, desc_rot0], dim=0)
            if max_id == 2:
                descriptorsAsall[:, i] = torch.concat([desc_rot2, desc_rot3, desc_rot0, desc_rot1], dim=0)
            if max_id == 3:
                descriptorsAsall[:, i] = torch.concat([desc_rot3, desc_rot0, desc_rot1, desc_rot2], dim=0)



        for i in range(descriptorsBsall.shape[1]):
            desc_rot0 = descriptorsBsall[0*sdim:1*sdim, i]
            desc_rot1 = descriptorsBsall[1*sdim:2*sdim, i]
            desc_rot2 = descriptorsBsall[2*sdim:3*sdim, i]
            desc_rot3 = descriptorsBsall[3*sdim:4*sdim, i]

            norm_res = torch.tensor([desc_rot0.sum(), desc_rot1.sum(), desc_rot2.sum(), desc_rot3.sum()])

            max_id = torch.argmax(norm_res)

            if max_id == 1:
                descriptorsBsall[:, i] = torch.concat([desc_rot1, desc_rot2, desc_rot3, desc_rot0], dim=0)
            if max_id == 2:
                descriptorsBsall[:, i] = torch.concat([desc_rot2, desc_rot3, desc_rot0, desc_rot1], dim=0)
            if max_id == 3:
                descriptorsBsall[:, i] = torch.concat([desc_rot3, desc_rot0, desc_rot1, desc_rot2], dim=0)



        # for i in range(descriptorsBsall.shape[1]):
        #     desc_rot0 = descriptorsBsall[np.arange(0, 4096, 4), i].clone()
        #     desc_rot1 = descriptorsBsall[np.arange(1, 4096, 4), i].clone()
        #     desc_rot2 = descriptorsBsall[np.arange(2, 4096, 4), i].clone()
        #     desc_rot3 = descriptorsBsall[np.arange(3, 4096, 4), i].clone()

        #     norm_res = torch.tensor([desc_rot0.norm(), desc_rot1.norm(), desc_rot2.norm(), desc_rot3.norm()])

        #     max_id = torch.argmax(norm_res)

        #     if max_id == 1:
        #         descriptorsBsall[np.arange(0, 4096, 4), i] = desc_rot1
        #         descriptorsBsall[np.arange(1, 4096, 4), i] = desc_rot2
        #         descriptorsBsall[np.arange(2, 4096, 4), i] = desc_rot3
        #         descriptorsBsall[np.arange(3, 4096, 4), i] = desc_rot0
        #     if max_id == 2:
        #         descriptorsBsall[np.arange(0, 4096, 4), i] = desc_rot2
        #         descriptorsBsall[np.arange(1, 4096, 4), i] = desc_rot3
        #         descriptorsBsall[np.arange(2, 4096, 4), i] = desc_rot0
        #         descriptorsBsall[np.arange(3, 4096, 4), i] = desc_rot1
        #     if max_id == 3:
        #         descriptorsBsall[np.arange(0, 4096, 4), i] = desc_rot3
        #         descriptorsBsall[np.arange(1, 4096, 4), i] = desc_rot0
        #         descriptorsBsall[np.arange(2, 4096, 4), i] = desc_rot1
        #         descriptorsBsall[np.arange(3, 4096, 4), i] = desc_rot2



        matches, scores = dense_feature_matching(descriptorsAsall, descriptorsBsall, 1, True)

        pointsA = pointsAall[matches[:, 0]]
        pointsB = pointsBall[matches[:, 1]]

        pointsA = np.roll(pointsA, 1, axis=-1)
        pointsB = np.roll(pointsB, 1, axis=-1)


        return pointsA, pointsB












    def transform(self, img):
        
        '''
        Convert given uint8 numpy array to tensor, perform normalization and 
        pad right/bottom to make image canvas a multiple of self.padding_n

        Parameters
        ----------
        img : nnumpy array (uint8)

        Returns
        -------
        img_T : torch.tensor
        (pad_right, pad_bottom) : int tuple 

        '''
        
        # transform to tensor and normalize
        T = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.to(self.device)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                                ])
        
        # zero padding to make image canvas a multiple of padding_n
        pad_right = 16 - img.shape[1] % self.padding_n if img.shape[1] % self.padding_n else 0
        pad_bottom = 16 - img.shape[0] % self.padding_n if img.shape[0] % self.padding_n else 0
        
        padding = torch.nn.ZeroPad2d([0, pad_right, 0, pad_bottom])
        
        # convert image
        #img_T = padding(T(img.astype(np.float32))).unsqueeze(0)
        img_T = padding(T(img)).unsqueeze(0)

        return img_T, (pad_right, pad_bottom)  
    
    @classmethod  
    def plot_keypoints(cls, img, pts, title='untitled', *args):
    
        f,a = plt.subplots()
        if len(args) > 0:
            pts2 = args[0]
            a.plot(pts2[0, :], pts2[1, :], marker='o', linestyle='none', color='green')
        
        a.plot(pts[0, :], pts[1, :], marker='+', linestyle='none', color='red')
        a.imshow(img)
        a.title.set_text(title)
        plt.pause(0.001)
        #plt.show() 








def dense_feature_matching(map_A, map_B, ratio_th, bidirectional=True):
    d1 = map_A.t()
    d1 /= torch.sqrt(torch.sum(torch.square(d1), 1)).unsqueeze(1)
    
    d2 = map_B.t()
    d2 /= torch.sqrt(torch.sum(torch.square(d2), 1)).unsqueeze(1)

    # perform matching
    matches, scores = mnn_ratio_matcher(d1, d2, ratio_th, bidirectional)

    return matches, scores



def mnn_ratio_matcher(descriptors1, descriptors2, ratio=0.8, bidirectional = True):
    
    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN and match similarity.
    nn12 = nns[:, 0]
    match_sim = nns_sim[:, 0]

    # Retrieve top 2 nearest neighbors 1->2.
    nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
    nns_dist = 2 - 2 * nns_sim
    # Compute Lowe's ratio.
    ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    # Save first NN.
    nn21 = nns[:, 0]
    
    # if not bidirectional, do not use ratios from 2 to 1
    ratios21[:] *= 1 if bidirectional else 0
        
    # Mutual NN + symmetric ratio test.
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)) # discard ratios21 to get the same results with matlab
    # Final matches.
    matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)
    match_sim = match_sim[mask]

    return (matches.data.cpu().numpy(),match_sim.data.cpu().numpy())













class Vgg19(torch.nn.Module):
    
    # modified from the original @ https://github.com/chenyuntc/pytorch-book/blob/master/chapter08-neural_style/PackedVGG.py
    
    def __init__(self, batch_normalization = True, required_layers = None):
        
        if required_layers == None and batch_normalization:
            required_layers = [3, 10, 17, 30, 43, 46]
        elif required_layers == None:
            required_layers = [2, 7, 12, 21, 30, 32]
            
        # features 2，7，12，21, 30, 32: conv1_2,conv2_2,relu3_2,relu4_2,conv5_2,conv5_3
        super(Vgg19, self).__init__()
        
        if batch_normalization:
            features = list(models.vgg19_bn(pretrained = True).features)[:47] # get vgg features
        else:
            features = list(models.vgg19(pretrained = True).features)[:33] # get vgg features
        
        self.features = torch.nn.ModuleList(features).eval() # construct network in eval mode
        
        for param in self.features.parameters(): # we don't need graidents, turn them of to save memory
            param.requires_grad = False
                
        self.required_layers = required_layers # record required layers to save them in forward
        
        for layer in required_layers[:-1]:
            self.features[layer+1].inplace = False # do not overwrite in any layer
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in self.required_layers:
                results.append(x)
                
        vgg_outputs = namedtuple("VggOutputs", ['conv1_2', 'conv2_2', 'relu3_2', 'relu4_2', 'conv5_2', 'conv5_3'])
        
        return vgg_outputs(*results)







    





def refine_points(points_A: torch.Tensor, points_B: torch.Tensor, activations_A: torch.Tensor, activations_B: torch.Tensor, ratio_th = 0.9, bidirectional = True):

    # normalize and reshape feature maps
    d1 = activations_A.squeeze(0) / activations_A.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
    d2 = activations_B.squeeze(0) / activations_B.squeeze(0).square().sum(0).sqrt().unsqueeze(0)
        
    # get number of points
    ch = d1.size(0)
    num_input_points = points_A.size(1)
    
    if num_input_points == 0:
        return points_A, points_B
    
    # upsample points
    points_A *= 2
    points_B *= 2
       
    # neighborhood to search
    neighbors = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # allocate space for scores
    scores = torch.zeros(num_input_points, neighbors.size(0), neighbors.size(0))
    
    # for each point search the refined matches in given [finer] resolution
    for i, n_A in enumerate(neighbors):   
        for j, n_B in enumerate(neighbors):

            # get features in the given neighborhood
            act_A = d1[:, points_A[1, :] + n_A[1], points_A[0, :] + n_A[0]].view(ch, -1)
            act_B = d2[:, points_B[1, :] + n_B[1], points_B[0, :] + n_B[0]].view(ch, -1)

            # compute mse
            scores[:, i, j] = torch.sum(act_A * act_B, 0)
            
    # retrieve top 2 nearest neighbors from A2B
    score_A, match_A = torch.topk(scores, 2, dim=2)
    score_A = 2 - 2 * score_A
    
    # compute lowe's ratio
    ratio_A2B = score_A[:, :, 0] / (score_A[:, :, 1] + 1e-8)
    
    # select the best match
    match_A2B = match_A[:, :, 0]
    score_A2B = score_A[:, :, 0]
    
    # retrieve top 2 nearest neighbors from B2A
    score_B, match_B = torch.topk(scores.transpose(2,1), 2, dim=2)
    score_B = 2 - 2 * score_B
    
    # compute lowe's ratio
    ratio_B2A = score_B[:, :, 0] / (score_B[:, :, 1] + 1e-8)
    
    # select the best match
    match_B2A = match_B[:, :, 0]
    #score_B2A = score_B[:, :, 0]
    
    # check for unique matches and apply ratio test
    ind_A = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_A2B).flatten()
    ind_B = (torch.arange(num_input_points).unsqueeze(1) * neighbors.size(0) + match_B2A).flatten()
    
    ind = torch.arange(num_input_points * neighbors.size(0))
    
    # if not bidirectional, do not use ratios from B to A
    ratio_B2A[:] *= 1 if bidirectional else 0 # discard ratio21 to get the same results with matlab
         
    mask = torch.logical_and(torch.max(ratio_A2B, ratio_B2A) < ratio_th,  (ind_B[ind_A] == ind).view(num_input_points, -1))
    
    # set a large SSE score for mathces above ratio threshold and not on to one (score_A2B <=4 so use 5)
    score_A2B[~mask] = 5
    
    # each input point can generate max two output points, so discard the two with highest SSE 
    _, discard = torch.topk(score_A2B, 2, dim=1)
    
    mask[torch.arange(num_input_points), discard[:, 0]] = 0
    mask[torch.arange(num_input_points), discard[:, 1]] = 0
    
    # x & y coordiates of candidate match points of A
    x = points_A[0, :].repeat(4, 1).t() + neighbors[:, 0].repeat(num_input_points, 1)
    y = points_A[1, :].repeat(4, 1).t() + neighbors[: ,1].repeat(num_input_points, 1)
    
    refined_points_A = torch.stack((x[mask], y[mask]))
    
    # x & y coordiates of candidate match points of A
    x = points_B[0, :].repeat(4, 1).t() + neighbors[:, 0][match_A2B]
    y = points_B[1, :].repeat(4, 1).t() + neighbors[:, 1][match_A2B]
    
    refined_points_B = torch.stack((x[mask], y[mask]))
        
    # if the number of refined matches is not enough to estimate homography,
    # but number of initial matches is enough, use initial points
    if refined_points_A.shape[1] < 4 and num_input_points > refined_points_A.shape[1]:
        refined_points_A = points_A
        refined_points_B = points_B
    
    return refined_points_A, refined_points_B




