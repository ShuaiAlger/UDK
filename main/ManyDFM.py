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


from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
from torchvision.models.efficientnet import efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models.efficientnet import efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s
# from torchvision.models.maxvit import maxvit_t
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

import pydegensac

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


class ManyDeepFeatureMatcher(torch.nn.Module):
    def __init__(self, model: str = 'None', device = None, bidirectional=True, enable_two_stage = True,
                 ratio_th = [0.9, 0.9, 0.9, 0.9, 0.95, 1.0], layers = []):
        super(ManyDeepFeatureMatcher, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device == None else device        


        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)
        self.padding_n = 16

        self.model_name = model

        self.return_nodes = {
        }
        for lid in range(len(layers)):
            self.return_nodes.update({layers[lid][0]: layers[lid][0]})

        nums = []
        for lid in range(len(layers)):
            # for in_lid in range(len(layers[lid])):
            nums.append(len(layers[lid]))
        print(nums)


        self.smodel = eval(self.model_name)(pretrained=True).to(self.device)
        # print(self.smodel)
        self.model = create_feature_extractor(self.smodel, return_nodes=self.return_nodes).to(self.device)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.padding_n = 16
        print('model is loaded.')


        self.enable_two_stage = enable_two_stage
        self.bidirectional = bidirectional
        self.ratio_th = np.array(ratio_th)

    def match(self, img_A, img_B, display_results=0, *args):

        maphA, mapwA = img_A.shape[0], img_A.shape[1]
        maphB, mapwB = img_B.shape[0], img_B.shape[1]


        block_size = 3
        sobel_size = 3
        k = 0.06

        grayA = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
        grayA = np.float32(grayA)
        corners_imgA = cv2.cornerHarris(grayA, block_size, sobel_size, k)

        grayB = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)
        grayB = np.float32(grayB)
        corners_imgB = cv2.cornerHarris(grayB, block_size, sobel_size, k)



        # transform into pytroch tensor and pad image to a multiple of 16
        inp_A, padding_A = self.transform(img_A) 
        inp_B, padding_B = self.transform(img_B)

        with torch.no_grad():

            outputs_A = self.model(inp_A)
            outputs_B = self.model(inp_B)

            activations_A = []
            activations_B = []


            for x in self.return_nodes:
                activations_A.append(outputs_A[str(x)])
                activations_B.append(outputs_B[str(x)])


            # for x in self.return_nodes:
            #     activations_A.append(outputs_A[str(x)].int_repr()/255.)
            #     activations_B.append(outputs_B[str(x)].int_repr()/255.)

            for i in range(len(activations_A)):
                activations_A[i] = self.upsample(activations_A[i])
                activations_B[i] = self.upsample(activations_B[i])

            maph, mapw = activations_A[0].shape[2], activations_A[0].shape[3]
            acA = activations_A[-4][0, :].cpu().numpy()
            acA = np.sum(acA, axis=0)
            acA = cv2.normalize(acA, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
            acA = cv2.resize(acA, (mapw, maph))
            # detectionA3 = cv2.applyColorMap(acA, cv2.COLORMAP_JET)
            
            maph, mapw = activations_B[0].shape[2], activations_B[0].shape[3]

            acB = activations_B[-4][0, :].cpu().numpy()
            acB = np.sum(acB, axis=0)


            acB = cv2.normalize(acB, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
            acB = cv2.resize(acB, (mapw, maph))
            # detectionB3 = cv2.applyColorMap(acB, cv2.COLORMAP_JET)

            detectionA3 = np.zeros_like(acA, dtype=np.float64)
            detectionB3 = np.zeros_like(acB, dtype=np.float64)


            if 1:
                detectionA3[acA < np.percentile(acA, 3)] = 1
                detectionB3[acB < np.percentile(acB, 3)] = 1

                detectionA3[acA > np.percentile(acA, 99.5)] = 1
                detectionB3[acB > np.percentile(acB, 99.5)] = 1

            else:
                corners_imgA = cv2.resize(corners_imgA, (detectionA3.shape[1], detectionA3.shape[0]))
                corners_imgB = cv2.resize(corners_imgB, (detectionB3.shape[1], detectionB3.shape[0]))

                detectionA3[corners_imgA > np.percentile(corners_imgA, 99)] = 1
                detectionB3[corners_imgB > np.percentile(corners_imgB, 99)] = 1



            detectionA = np.argwhere(detectionA3 > 0)
            detectionB = np.argwhere(detectionB3 > 0)

            descriptorsA = activations_A[0][0][:, detectionA[:, 0], detectionA[:, 1]]
            descriptorsB = activations_B[0][0][:, detectionB[:, 0], detectionB[:, 1]]


            descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)

            descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:, np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)

            descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:256, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
            descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:256, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)


            # descriptorsA = torch.concat([descriptorsA, activations_A[1][0][0:64, np.int64(detectionA[:, 0]/2), np.int64(detectionA[:, 1]/2)]], dim=0)
            # descriptorsB = torch.concat([descriptorsB, activations_B[1][0][0:64, np.int64(detectionB[:, 0]/2), np.int64(detectionB[:, 1]/2)]], dim=0)

            # descriptorsA = torch.concat([descriptorsA, activations_A[2][0][0:128, np.int64(detectionA[:, 0]/4), np.int64(detectionA[:, 1]/4)]], dim=0)
            # descriptorsB = torch.concat([descriptorsB, activations_B[2][0][0:128, np.int64(detectionB[:, 0]/4), np.int64(detectionB[:, 1]/4)]], dim=0)

            # descriptorsA = torch.concat([descriptorsA, activations_A[3][0][0:128, np.int64(detectionA[:, 0]/8), np.int64(detectionA[:, 1]/8)]], dim=0)
            # descriptorsB = torch.concat([descriptorsB, activations_B[3][0][0:128, np.int64(detectionB[:, 0]/8), np.int64(detectionB[:, 1]/8)]], dim=0)


            # print(detectionA3.shape, detectionB3.shape, detectionA.shape, detectionB.shape, descriptorsA.shape, descriptorsB.shape)



            matches, scores = dense_feature_matching(descriptorsA, descriptorsB, 1, True)



            pointsA = detectionA[matches[:, 0]]
            pointsB = detectionB[matches[:, 1]]

            # print(pointsA[:, 0].max(), pointsA[:, 1].max(), pointsB[:, 0].max(), pointsB[:, 1].max())

            pointsA = np.roll(pointsA, 1, axis=-1)
            pointsB = np.roll(pointsB, 1, axis=-1)



            canvas = draw_green(img_A, img_B, pointsA, pointsB)



            detectionA3 = cv2.normalize(detectionA3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
            detectionB3 = cv2.normalize(detectionB3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8UC1)
            detectionA3 = cv2.cvtColor(detectionA3, cv2.COLOR_GRAY2BGR)
            detectionB3 = cv2.cvtColor(detectionB3, cv2.COLOR_GRAY2BGR)

            img_A = cv2.resize(img_A, (detectionA3.shape[1], detectionA3.shape[0]))
            img_B = cv2.resize(img_B, (detectionB3.shape[1], detectionB3.shape[0]))



            # if USE_CASCADE_RANSAC:
            #     if len(pointsA) > 8:
            #         model, inliers = pydegensac.findFundamentalMatrix(pointsA, pointsB, 3.0, 0.99, 10000)
            #         pointsA, pointsB = pointsA[inliers], pointsB[inliers]




            canvas2 = draw_green(img_A*detectionA3, img_B*detectionB3, pointsA, pointsB)


            cv2.namedWindow("canvas", 0)
            cv2.imshow("canvas", canvas)

            cv2.namedWindow("canvas2", 0)
            cv2.imshow("canvas2", canvas2)

            cv2.waitKey(1)






        return pointsA, pointsB



    def transform(self, img):

        '''
        Convert given uint8 numpy array to tensor, perform normalization and 
        pad right/bottom to make image canvas a multiple of self.padding_n

        Parameters
        ----------
        img : numpy array (uint8)

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

