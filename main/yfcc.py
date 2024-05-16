from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv


from sg_eval_utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc,
                          read_image, read_image_color,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


from pathlib import Path

import cv2


class YFCCDataset(utils.data.Dataset):
    def __init__(self,
                 input_pairs_txt = '/home/shuai/Documents/DRAEM/assets/yfcc_test_pairs_with_gt.txt',
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()


        self.mode = mode

        self.input_dir = '/media/shuai/Correspondence/DATASETS/YFCC/raw_data/yfcc100m/'
        
        with open(input_pairs_txt, 'r') as f:
            self.pairs = [l.split() for l in f.readlines()]

            # for i, pair in enumerate(pairs):
            #     name0, name1 = pair[:2]
            #     stem0, stem1 = Path(name0).stem, Path(name1).stem

            #     # If a rotation integer is provided (e.g. from EXIF data), use it:
            #     if len(pair) >= 5:
            #         rot0, rot1 = int(pair[2]), int(pair[3])
            #     else:
            #         rot0, rot1 = 0, 0

            #     # Load the image pair.
            #     image0, inp0, scales0 = read_image(
            #         input_dir / name0, 'cuda', resize=(640, 480), rotation=rot0, resize_float=False)
            #     image1, inp1, scales1 = read_image(
            #         input_dir / name1, 'cuda', resize=(640, 480), rotation=rot1, resize_float=False)

            #     if image0 is None or image1 is None:
            #         print('Problem reading image pair: {} {}'.format(
            #             input_dir/name0, input_dir/name1))
            #         exit(1)

            #     assert len(pair) == 38, 'Pair does not have ground truth info'
            #     K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            #     K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            #     T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            #     # Scale the intrinsics to resized image.
            #     K0 = scale_intrinsics(K0, scales0)
            #     K1 = scale_intrinsics(K1, scales1)

            #     # Update the intrinsics + extrinsics if EXIF rotation was found.
            #     if rot0 != 0 or rot1 != 0:
            #         cam0_T_w = np.eye(4)
            #         cam1_T_w = T_0to1
            #         if rot0 != 0:
            #             K0 = rotate_intrinsics(K0, image0.shape, rot0)
            #             cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            #         if rot1 != 0:
            #             K1 = rotate_intrinsics(K1, image1.shape, rot1)
            #             cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            #         cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            #         T_0to1 = cam1_T_cam0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        pair = self.pairs[idx]
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem

        scene_name = stem0 + stem1

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0


        # image0, inp0, scales0 = read_image(
        #     self.input_dir +"/"+ name0, 'cpu', resize=(640, 480), rotation=rot0, resize_float=False)
        # image1, inp1, scales1 = read_image(
        #     self.input_dir +"/"+ name1, 'cpu', resize=(640, 480), rotation=rot1, resize_float=False)

        image0, inp0, scales0 = read_image_color(
            self.input_dir +"/"+ name0, 'cpu', resize=(640, 480), rotation=rot0, resize_float=False)
        image1, inp1, scales1 = read_image_color(
            self.input_dir +"/"+ name1, 'cpu', resize=(640, 480), rotation=rot1, resize_float=False)

        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                self.input_dir/name0, self.input_dir/name1))
            exit(1)

        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0



        K0 = torch.tensor(K0.copy(), dtype=torch.float32).reshape(3, 3)
        K1 = torch.tensor(K1.copy(), dtype=torch.float32).reshape(3, 3)

        T_0to1 = torch.tensor(T_0to1.copy(), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()


        data = {
            'image0': image0,   # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K0,  # (3, 3)
            'K1': K1,
            'dataset_name': 'YFCC',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem1}.jpg'))
        }

        return data







class YFCCDatasetRot90(utils.data.Dataset):
    def __init__(self,
                 input_pairs_txt = '/home/shuai/Documents/DRAEM/assets/yfcc_test_pairs_with_gt.txt',
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()


        self.mode = mode

        self.input_dir = '/media/shuai/Correspondence/DATASETS/YFCC/raw_data/yfcc100m/'
        
        with open(input_pairs_txt, 'r') as f:
            self.pairs = [l.split() for l in f.readlines()]

            # for i, pair in enumerate(pairs):
            #     name0, name1 = pair[:2]
            #     stem0, stem1 = Path(name0).stem, Path(name1).stem

            #     # If a rotation integer is provided (e.g. from EXIF data), use it:
            #     if len(pair) >= 5:
            #         rot0, rot1 = int(pair[2]), int(pair[3])
            #     else:
            #         rot0, rot1 = 0, 0

            #     # Load the image pair.
            #     image0, inp0, scales0 = read_image(
            #         input_dir / name0, 'cuda', resize=(640, 480), rotation=rot0, resize_float=False)
            #     image1, inp1, scales1 = read_image(
            #         input_dir / name1, 'cuda', resize=(640, 480), rotation=rot1, resize_float=False)

            #     if image0 is None or image1 is None:
            #         print('Problem reading image pair: {} {}'.format(
            #             input_dir/name0, input_dir/name1))
            #         exit(1)

            #     assert len(pair) == 38, 'Pair does not have ground truth info'
            #     K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            #     K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            #     T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            #     # Scale the intrinsics to resized image.
            #     K0 = scale_intrinsics(K0, scales0)
            #     K1 = scale_intrinsics(K1, scales1)

            #     # Update the intrinsics + extrinsics if EXIF rotation was found.
            #     if rot0 != 0 or rot1 != 0:
            #         cam0_T_w = np.eye(4)
            #         cam1_T_w = T_0to1
            #         if rot0 != 0:
            #             K0 = rotate_intrinsics(K0, image0.shape, rot0)
            #             cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            #         if rot1 != 0:
            #             K1 = rotate_intrinsics(K1, image1.shape, rot1)
            #             cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            #         cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            #         T_0to1 = cam1_T_cam0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        
        pair = self.pairs[idx]
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem

        scene_name = stem0 + stem1

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        
        USE_ROT90_DATASET = 1
        rot0 = 0
        rot1 = 0


        image0, inp0, scales0 = read_image_color(
            self.input_dir +"/"+ name0, 'cpu', resize=(640, 480), rotation=rot0, resize_float=False)
        image1, inp1, scales1 = read_image_color(
            self.input_dir +"/"+ name1, 'cpu', resize=(640, 480), rotation=rot1, resize_float=False)



        if USE_ROT90_DATASET:
            random_rotation = np.random.random()
            if 0 < random_rotation and random_rotation <= 0.25:
                rot1 = 0
            if 0.25 < random_rotation and random_rotation <= 0.5:
                rot1 = 1 
            if 0.5 < random_rotation and random_rotation <= 0.75:
                rot1 = 2
            if 0.75 < random_rotation and random_rotation <= 1.0:
                rot1 = 3

            if rot1 == 1:
                image1 = cv2.rotate(image1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if rot1 == 2:
                image1 = cv2.rotate(image1, cv2.ROTATE_180)
            if rot1 == 3:
                image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)



        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                self.input_dir/name0, self.input_dir/name1))
            exit(1)

        assert len(pair) == 38, 'Pair does not have ground truth info'
        K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
        K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
        T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

        # Scale the intrinsics to resized image.
        K0 = scale_intrinsics(K0, scales0)
        K1 = scale_intrinsics(K1, scales1)

        # Update the intrinsics + extrinsics if EXIF rotation was found.
        if rot0 != 0 or rot1 != 0:
            cam0_T_w = np.eye(4)
            cam1_T_w = T_0to1
            if rot0 != 0:
                K0 = rotate_intrinsics(K0, image0.shape, rot0)
                cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
            if rot1 != 0:
                K1 = rotate_intrinsics(K1, image1.shape, rot1)
                cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
            cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
            T_0to1 = cam1_T_cam0



        K0 = torch.tensor(K0.copy(), dtype=torch.float32).reshape(3, 3)
        K1 = torch.tensor(K1.copy(), dtype=torch.float32).reshape(3, 3)

        T_0to1 = torch.tensor(T_0to1.copy(), dtype=torch.float32)
        T_1to0 = T_0to1.inverse()


        data = {
            'image0': image0,   # (1, h, w)
            'image1': image1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K0,  # (3, 3)
            'K1': K1,
            'dataset_name': 'YFCC',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, 'color', f'{stem0}.jpg'),
                           osp.join(scene_name, 'color', f'{stem1}.jpg'))
        }

        return data
    



