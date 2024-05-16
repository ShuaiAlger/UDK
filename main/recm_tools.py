
from copy import deepcopy
import cv2,math
import numpy as np
import torch
from torch import nn

torch.set_grad_enabled(False)


def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x, np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def rotate_image_bound_with_M(image, angle):
    if angle == 0:
        return image,np.array([[1,0,0],[0,1,0]])

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int(0.5 + (h * sin) + (w * cos))
    nH = int(0.5 + (h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_AREA, borderValue=(255, 255, 255)),M

def rotate_image_bound(image, angle):
    img, M = rotate_image_bound_with_M(image, angle)
    return img

def calRotateAngleFromMatch(mkpts0, mkpts1):
    mat = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    return calRotateAngleFromMatrix(mat)

def calTransformationFromMatch(mkpts0, mkpts1):
    mat = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
    return mat[0][:,2].T

def calRotateAngleFromMatrix(mat):
    rmat = mat[0][:2, :2]
    det = np.linalg.det(rmat)
    rmat_normal = rmat / (det ** 0.5)
    angle = math.asin(rmat_normal[1, 0]) * 180 / math.pi
    return angle



if __name__ == '__main__':

    img = cv2.imread('./test.png')

    img_r45, M = rotate_image_bound_with_M(img, 45)
    M = np.row_stack((M, np.array([0, 0, 1])))
    M_inv = np.mat(np.linalg.inv(M))


    mkpts0, mkpts1 = [], []


    # # unproject points
    # hmkpts0 = add_ones(mkpts0)
    # rhmkpts0 = (M_inv * hmkpts0.T).A.T[:, 0:2]

